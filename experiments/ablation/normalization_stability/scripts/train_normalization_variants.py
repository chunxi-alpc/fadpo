#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional


VARIANTS = ("uniform", "no_normalization", "original_normalized", "capped_normalization")


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - server dependency check
        raise RuntimeError("Install PyYAML or provide a JSON config file.") from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def resolve_path(config_path: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def load_runtime_dependencies():
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import DPOConfig, DPOTrainer
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    return {
        "torch": torch,
        "F": F,
        "Accelerator": Accelerator,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "pad_sequence": pad_sequence,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "TrainerCallback": TrainerCallback,
        "set_seed": set_seed,
        "DPOConfig": DPOConfig,
        "DPOTrainer": DPOTrainer,
        "DataCollatorForPreference": DataCollatorForPreference,
    }


def build_classes(deps: Dict[str, Any]):
    torch = deps["torch"]
    F = deps["F"]
    pad_sequence = deps["pad_sequence"]
    TrainerCallback = deps["TrainerCallback"]
    DPOTrainer = deps["DPOTrainer"]
    DataCollatorForPreference = deps["DataCollatorForPreference"]
    Accelerator = deps["Accelerator"]

    @dataclass
    class NormalizationDataCollator(DataCollatorForPreference):
        mask_field: str = "topo_mask"
        use_margin: bool = True

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            token_masks = None
            if features and self.mask_field in features[0]:
                token_masks = [f.pop(self.mask_field) for f in features]
            margins = None
            if features and self.use_margin and "margin" in features[0]:
                margins = [f.pop("margin") for f in features]

            batch = super().__call__(features)

            if token_masks is not None:
                mask_tensors = [torch.tensor(mask, dtype=torch.float32) for mask in token_masks]
                batch["raw_token_weights"] = pad_sequence(mask_tensors, batch_first=True, padding_value=0.0)

                rejected_len = batch["rejected_input_ids"].shape[1]
                mask_len = batch["raw_token_weights"].shape[1]
                if mask_len < rejected_len:
                    batch["raw_token_weights"] = F.pad(
                        batch["raw_token_weights"], (0, rejected_len - mask_len), value=0.0
                    )
                elif mask_len > rejected_len:
                    batch["raw_token_weights"] = batch["raw_token_weights"][:, :rejected_len]

            if margins is not None:
                batch["margin"] = torch.tensor(margins, dtype=torch.float32)

            return batch

    class JsonlMetricsCallback(TrainerCallback):
        def __init__(self, output_path: Path, variant: str):
            self.output_path = output_path
            self.variant = variant
            self.last_grad_norm: float | None = None

        def _compute_grad_norm(self, model: Any) -> float | None:
            if model is None:
                return None
            total_sq = 0.0
            found = False
            for parameter in model.parameters():
                grad = parameter.grad
                if grad is None:
                    continue
                grad_norm = grad.detach().float().norm(2)
                if not torch.isfinite(grad_norm):
                    return float("nan")
                total_sq += float(grad_norm.item()) ** 2
                found = True
            if not found:
                return None
            return math.sqrt(total_sq)

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):  # noqa: D401
            self.last_grad_norm = self._compute_grad_norm(model)

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
            if not getattr(state, "is_world_process_zero", True):
                return
            if not logs:
                return
            row: Dict[str, Any] = {"step": int(state.global_step), "variant": self.variant}
            wanted = (
                "loss",
                "total_loss",
                "main_loss",
                "grad_norm",
                "learning_rate",
                "max_effective_weight",
                "mean_effective_weight",
                "high_risk_token_ratio",
                "high_risk_effective_weight_median",
                "high_risk_effective_weight_p95",
                "high_risk_effective_weight_max",
                "high_risk_effective_weight_count",
                "active_token_count",
                "effective_weight_sum",
                "nan_or_inf",
            )
            for key in wanted:
                if key in logs:
                    row[key] = logs[key]
            if "grad_norm" not in row and self.last_grad_norm is not None:
                row["grad_norm"] = self.last_grad_norm
            if "nan_or_inf" not in row:
                numeric_values = [finite_or_none(value) for value in row.values()]
                row["nan_or_inf"] = any(value is None for value in numeric_values if value is not None)
            append_jsonl(self.output_path, row)

    class NormalizationDPOTrainer(DPOTrainer):
        def __init__(
            self,
            model,
            ref_model=None,
            args=None,
            variant: str = "original_normalized",
            cap: float = 10.0,
            high_risk_threshold: float = 0.5,
            margin_scale_train: float = 0.2,
            use_margin: bool = True,
            **kwargs,
        ):
            if variant not in VARIANTS:
                raise ValueError(f"Unsupported variant: {variant}")
            if args is not None and getattr(args, "gradient_checkpointing", False):
                if args.gradient_checkpointing_kwargs is None:
                    args.gradient_checkpointing_kwargs = {}
                if args.gradient_checkpointing_kwargs.get("use_reentrant", True):
                    if Accelerator().is_local_main_process:
                        print("Forcing gradient_checkpointing use_reentrant=False for DDP compatibility.")
                    args.gradient_checkpointing_kwargs["use_reentrant"] = False
            self.variant = variant
            self.cap = cap
            self.high_risk_threshold = high_risk_threshold
            self.margin_scale_train = margin_scale_train
            self.use_margin = use_margin
            super().__init__(model, ref_model=ref_model, args=args, **kwargs)

        def _align_raw_weights(
            self,
            raw_weights: Any,
            prompt_lens: Any,
            target_shape: Any,
        ):
            aligned = torch.zeros(target_shape, dtype=torch.float32, device=self.accelerator.device)
            for idx in range(target_shape[0]):
                prompt_len = int(prompt_lens[idx].item())
                start_idx = max(0, prompt_len - 1)
                content = raw_weights[idx]
                fill_len = min(content.shape[0], target_shape[1] - start_idx)
                if fill_len > 0:
                    aligned[idx, start_idx : start_idx + fill_len] = content[:fill_len]
            return aligned

        def _effective_rejected_weights(self, raw_weights: Any, rejected_mask: Any):
            mask_f = rejected_mask.float()
            active_count = mask_f.sum(-1).clamp_min(1.0)
            raw = raw_weights.clamp_min(0.0) * mask_f

            if self.variant == "uniform":
                effective = mask_f
            elif self.variant == "no_normalization":
                effective = raw
            else:
                raw_sum = raw.sum(-1, keepdim=True)
                uniform_fallback = mask_f
                normalized = raw * (active_count.view(-1, 1) / raw_sum.clamp_min(1e-8))
                normalized = torch.where(raw_sum > 0, normalized, uniform_fallback)
                if self.variant == "capped_normalization":
                    capped = normalized.clamp_max(self.cap) * mask_f
                    capped_sum = capped.sum(-1, keepdim=True)
                    effective = capped * (active_count.view(-1, 1) / capped_sum.clamp_min(1e-8))
                    effective = torch.where(capped_sum > 0, effective, uniform_fallback)
                else:
                    effective = normalized
            return effective * mask_f, raw, active_count

        def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Any],
            train_eval: Literal["train", "eval"] = "train",
        ):
            metrics: Dict[str, Any] = {}
            prefix = "eval_" if train_eval == "eval" else ""

            if "prompt_attention_mask" in batch:
                prompt_lens = batch["prompt_attention_mask"].sum(dim=1)
            else:
                prompt_lens = torch.tensor([0] * batch["chosen_input_ids"].shape[0], device=self.accelerator.device)

            def build_labels_per_sample(input_ids, attention_mask, p_lens):
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                for idx, prompt_len in enumerate(p_lens):
                    labels[idx, : int(prompt_len.item())] = -100
                return labels

            if "chosen_labels" not in batch:
                batch["chosen_labels"] = build_labels_per_sample(
                    batch["chosen_input_ids"], batch["chosen_attention_mask"], prompt_lens
                )
            if "rejected_labels" not in batch:
                batch["rejected_labels"] = build_labels_per_sample(
                    batch["rejected_input_ids"], batch["rejected_attention_mask"], prompt_lens
                )

            chosen_input_ids = batch["chosen_input_ids"]
            chosen_attention_mask = batch["chosen_attention_mask"]
            chosen_labels = batch["chosen_labels"]
            rejected_input_ids = batch["rejected_input_ids"]
            rejected_attention_mask = batch["rejected_attention_mask"]
            rejected_labels = batch["rejected_labels"]

            max_len = max(chosen_input_ids.shape[1], rejected_input_ids.shape[1])
            pad_token_id = self.padding_value if self.padding_value is not None else 0
            label_pad_token_id = -100

            def pad_to_len(tensor, length, pad_value):
                if tensor.shape[1] >= length:
                    return tensor
                return F.pad(tensor, (0, length - tensor.shape[1]), value=pad_value)

            c_ids = pad_to_len(chosen_input_ids, max_len, pad_token_id)
            c_mask = pad_to_len(chosen_attention_mask, max_len, 0)
            c_labels = pad_to_len(chosen_labels, max_len, label_pad_token_id)
            r_ids = pad_to_len(rejected_input_ids, max_len, pad_token_id)
            r_mask = pad_to_len(rejected_attention_mask, max_len, 0)
            r_labels = pad_to_len(rejected_labels, max_len, label_pad_token_id)

            input_ids = torch.cat([c_ids, r_ids], dim=0)
            attention_mask = torch.cat([c_mask, r_mask], dim=0)
            labels = torch.cat([c_labels, r_labels], dim=0)
            len_chosen = c_ids.shape[0]

            def get_logprobs(logits, labels):
                labels = labels[:, 1:].clone()
                logits = logits[:, :-1, :]
                loss_mask = labels != -100
                log_probs = F.log_softmax(logits, dim=-1)
                labels[labels == -100] = 0
                per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
                return per_token_logps, loss_mask

            with torch.no_grad():
                if self.ref_model is None:
                    unwrapped_model = model.module if hasattr(model, "module") else model
                    with unwrapped_model.disable_adapter():
                        ref_output = unwrapped_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                else:
                    ref_output = self.ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                all_ref_logps, _ = get_logprobs(ref_output.logits, labels)
                all_ref_logps = all_ref_logps.detach()

            model_output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            all_logps, all_masks = get_logprobs(model_output.logits, labels)

            chosen_logps_token = all_logps[:len_chosen]
            rejected_logps_token = all_logps[len_chosen:]
            chosen_mask = all_masks[:len_chosen]
            rejected_mask = all_masks[len_chosen:]
            ref_chosen_logps_token = all_ref_logps[:len_chosen]
            ref_rejected_logps_token = all_ref_logps[len_chosen:]

            if "raw_token_weights" in batch:
                raw_weights = batch["raw_token_weights"].to(self.accelerator.device)
                aligned_raw_weights = self._align_raw_weights(raw_weights, prompt_lens, rejected_logps_token.shape)
            else:
                aligned_raw_weights = torch.ones_like(rejected_logps_token, dtype=torch.float32)

            effective_weights, raw_active_weights, active_count = self._effective_rejected_weights(
                aligned_raw_weights, rejected_mask
            )

            chosen_token_logratios = (chosen_logps_token - ref_chosen_logps_token).float() * chosen_mask.float()
            rejected_token_logratios = (
                (rejected_logps_token - ref_rejected_logps_token).float() * rejected_mask.float()
            )
            chosen_delta = chosen_token_logratios.sum(-1)
            rejected_delta = (effective_weights * rejected_token_logratios).sum(-1)
            logits = chosen_delta - rejected_delta

            if self.use_margin and "margin" in batch:
                raw_margin = batch["margin"].to(self.accelerator.device, dtype=logits.dtype)
                margin = raw_margin * self.margin_scale_train
            else:
                raw_margin = torch.zeros_like(logits)
                margin = torch.zeros_like(logits)

            losses = -F.logsigmoid(self.beta * logits - margin)
            loss = losses.mean()

            active_denominator = active_count.clamp_min(1.0)
            high_risk_mask = (raw_active_weights > self.high_risk_threshold) & rejected_mask.bool()
            high_risk_ratio = high_risk_mask.float().sum(-1) / active_denominator
            max_effective_weight = effective_weights.max(dim=-1).values
            mean_effective_weight = effective_weights.sum(-1) / active_denominator
            high_risk_effective_values = effective_weights[high_risk_mask].float()
            if high_risk_effective_values.numel() > 0:
                high_risk_weight_median = torch.quantile(high_risk_effective_values, 0.50)
                high_risk_weight_p95 = torch.quantile(high_risk_effective_values, 0.95)
                high_risk_weight_max = high_risk_effective_values.max()
                high_risk_weight_count = torch.tensor(
                    float(high_risk_effective_values.numel()), device=self.accelerator.device
                )
            else:
                high_risk_weight_median = torch.tensor(0.0, device=self.accelerator.device)
                high_risk_weight_p95 = torch.tensor(0.0, device=self.accelerator.device)
                high_risk_weight_max = torch.tensor(0.0, device=self.accelerator.device)
                high_risk_weight_count = torch.tensor(0.0, device=self.accelerator.device)
            finite_loss = torch.isfinite(loss)

            metrics[f"{prefix}loss"] = loss.detach().cpu()
            metrics[f"{prefix}total_loss"] = loss.detach().cpu()
            metrics[f"{prefix}main_loss"] = loss.detach().cpu()
            metrics[f"{prefix}max_effective_weight"] = max_effective_weight.mean().detach().cpu()
            metrics[f"{prefix}mean_effective_weight"] = mean_effective_weight.mean().detach().cpu()
            metrics[f"{prefix}high_risk_token_ratio"] = high_risk_ratio.mean().detach().cpu()
            metrics[f"{prefix}high_risk_effective_weight_median"] = high_risk_weight_median.detach().cpu()
            metrics[f"{prefix}high_risk_effective_weight_p95"] = high_risk_weight_p95.detach().cpu()
            metrics[f"{prefix}high_risk_effective_weight_max"] = high_risk_weight_max.detach().cpu()
            metrics[f"{prefix}high_risk_effective_weight_count"] = high_risk_weight_count.detach().cpu()
            metrics[f"{prefix}active_token_count"] = active_count.mean().detach().cpu()
            metrics[f"{prefix}effective_weight_sum"] = effective_weights.sum(-1).mean().detach().cpu()
            metrics[f"{prefix}raw_weight_sum"] = raw_active_weights.sum(-1).mean().detach().cpu()
            metrics[f"{prefix}margin/raw_mean"] = raw_margin.mean().detach().cpu()
            metrics[f"{prefix}margin/scaled_mean"] = margin.mean().detach().cpu()
            metrics[f"{prefix}nan_or_inf"] = torch.tensor(0.0 if finite_loss else 1.0).cpu()

            with torch.no_grad():
                chosen_rewards = self.beta * chosen_delta
                rejected_rewards = self.beta * rejected_delta
                reward_accuracies = (chosen_rewards > rejected_rewards).float()
            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().detach().cpu()
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().detach().cpu()
            metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().detach().cpu()
            metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().detach().cpu()

            return loss, metrics

    return NormalizationDataCollator, JsonlMetricsCallback, NormalizationDPOTrainer


def variant_output_dir(runs_dir: Path, variant: str) -> Path:
    return runs_dir / variant


def train_variant(config_path: Path, config: Dict[str, Any], variant: str) -> None:
    deps = load_runtime_dependencies()
    torch = deps["torch"]
    load_dataset = deps["load_dataset"]
    LoraConfig = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    set_seed = deps["set_seed"]
    DPOConfig = deps["DPOConfig"]

    NormalizationDataCollator, JsonlMetricsCallback, NormalizationDPOTrainer = build_classes(deps)

    paths = config["paths"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    lora_cfg = config["lora"]
    weights_cfg = config["weights"]

    train_dataset_path = resolve_path(config_path, paths["train_dataset"])
    eval_dataset_path = resolve_path(config_path, paths["eval_dataset"])
    runs_dir = resolve_path(config_path, paths["runs_dir"])
    output_dir = variant_output_dir(runs_dir, variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    set_seed(int(training_cfg.get("seed", 42)))

    base_model = model_cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_dtype = torch.bfloat16 if bool(training_cfg.get("bf16", True)) else torch.float16
    model_kwargs = {
        "dtype": model_dtype,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
    }
    attn_implementation = model_cfg.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )
    model = get_peft_model(model, peft_config)

    train_dataset = load_dataset("json", data_files=str(train_dataset_path), split="train")
    eval_dataset = load_dataset("json", data_files=str(eval_dataset_path), split="train")

    training_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        eval_strategy="steps",
        logging_steps=int(training_cfg.get("logging_steps", 1)),
        eval_steps=int(training_cfg.get("eval_steps", 100)),
        save_strategy="steps",
        save_steps=int(training_cfg.get("save_steps", 100)),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.1)),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        bf16=bool(training_cfg.get("bf16", True)),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        max_length=int(training_cfg["max_length"]),
        max_prompt_length=int(training_cfg["max_prompt_length"]),
        truncation_mode="keep_end",
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=float(training_cfg["beta"]),
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs
        )
    model.enable_input_require_grads()
    model.config.use_cache = False

    data_collator = NormalizationDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        mask_field=str(weights_cfg.get("mask_field", "topo_mask")),
        use_margin=bool(training_cfg.get("use_margin", True)),
    )

    trainer = NormalizationDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        variant=variant,
        cap=float(weights_cfg.get("capped_normalization_cap", 10.0)),
        high_risk_threshold=float(weights_cfg.get("high_risk_threshold", 0.5)),
        margin_scale_train=float(training_cfg.get("margin_scale_train", 0.2)),
        use_margin=bool(training_cfg.get("use_margin", True)),
        callbacks=[JsonlMetricsCallback(metrics_path, variant)],
    )

    if trainer.is_world_process_zero():
        write_json(
            output_dir / "run_config.json",
            {
                "variant": variant,
                "config_path": str(config_path),
                "train_dataset": str(train_dataset_path),
                "eval_dataset": str(eval_dataset_path),
                "base_model": base_model,
                "training": training_cfg,
                "weights": weights_cfg,
            },
        )

    trainer.train()
    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
    metrics = trainer.evaluate()
    if trainer.is_world_process_zero():
        write_json(output_dir / "trainer_eval_metrics.json", metrics)


def selected_variants(config: Dict[str, Any], requested: str) -> Iterable[str]:
    configured = list(config.get("training", {}).get("variants", VARIANTS))
    if requested == "all":
        return configured
    if requested not in VARIANTS:
        raise ValueError(f"Unsupported variant: {requested}")
    if requested not in configured:
        return [requested]
    return [requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train normalization-stability Fa-DPO variants.")
    parser.add_argument(
        "--config",
        default="../configs/normalization_stability.yaml",
        help="Path to normalization_stability.yaml. Relative paths inside it are resolved from the config file.",
    )
    parser.add_argument(
        "--variant",
        default="all",
        choices=("all", *VARIANTS),
        help="Variant to run. Use all to train the configured sequence.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override model.base_model from the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (script_dir / config_path).resolve()
    config = load_yaml(config_path)
    if args.base_model:
        config.setdefault("model", {})["base_model"] = args.base_model
    for variant in selected_variants(config, args.variant):
        train_variant(config_path, config, variant)


if __name__ == "__main__":
    main()
