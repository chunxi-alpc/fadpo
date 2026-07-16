#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForPreference

from common import artifact_path, checkpoint_path, iso_utc_now, write_run_metadata


@dataclass
class TopoDPODataCollator(DataCollatorForPreference):
    use_topo_mask: bool = True
    use_margin: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        topo_masks = [f.pop("topo_mask") for f in features] if "topo_mask" in features[0] else None
        margins = [f.pop("margin") for f in features] if "margin" in features[0] else None
        if not self.use_topo_mask:
            topo_masks = None
        if not self.use_margin:
            margins = None

        batch = super().__call__(features)

        if topo_masks is not None:
            topo_masks_tensors = [torch.tensor(mask, dtype=torch.float32) for mask in topo_masks]
            batch["topo_mask"] = pad_sequence(topo_masks_tensors, batch_first=True, padding_value=0.0)

            rejected_len = batch["rejected_input_ids"].shape[1]
            topo_len = batch["topo_mask"].shape[1]
            if topo_len < rejected_len:
                batch["topo_mask"] = F.pad(batch["topo_mask"], (0, rejected_len - topo_len), value=0.0)
            elif topo_len > rejected_len:
                batch["topo_mask"] = batch["topo_mask"][:, :rejected_len]

        if margins is not None:
            batch["margin"] = torch.tensor(margins, dtype=torch.float32)

        return batch


class TopoDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model,
        ref_model=None,
        args=None,
        lambda_coeff=0.05,
        margin_scale_train=0.2,
        topo_threshold=0.5,
        **kwargs,
    ):
        if args is not None and args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                args.gradient_checkpointing_kwargs = {}
            if args.gradient_checkpointing_kwargs.get("use_reentrant", True):
                if Accelerator().is_local_main_process:
                    print("TopoDPOTrainer: Forcing use_reentrant=False to fix DDP error.")
                args.gradient_checkpointing_kwargs["use_reentrant"] = False

        self.lambda_coeff = lambda_coeff
        self.margin_scale_train = margin_scale_train
        self.topo_threshold = topo_threshold
        super().__init__(model, ref_model=ref_model, args=args, **kwargs)

    def _align_topo_mask(
        self,
        topo_mask: torch.Tensor,
        prompt_lens: torch.Tensor,
        target_shape: torch.Size,
    ) -> torch.Tensor:
        aligned_topo_mask = torch.zeros(target_shape, dtype=torch.float32, device=self.accelerator.device)
        for idx in range(target_shape[0]):
            prompt_len = int(prompt_lens[idx].item())
            start_idx = max(0, prompt_len - 1)
            raw_mask_content = topo_mask[idx]
            fill_len = min(raw_mask_content.shape[0], target_shape[1] - start_idx)
            if fill_len > 0:
                aligned_topo_mask[idx, start_idx : start_idx + fill_len] = raw_mask_content[:fill_len]
        return aligned_topo_mask

    def _compute_topo_aux_loss(
        self,
        aligned_topo_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
        rejected_logps_token: torch.Tensor,
        ref_rejected_logps_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rejected_mask_f = rejected_mask.float()
        weighted_topo_mask = aligned_topo_mask.clamp_min(0.0) * rejected_mask_f
        active_weight_sum = weighted_topo_mask.sum(-1)
        active_token_count = (weighted_topo_mask > 0).float().sum(-1)
        high_risk_token_count = ((aligned_topo_mask > self.topo_threshold) * rejected_mask_f).sum(-1)

        token_gap = self.beta * (rejected_logps_token.float() - ref_rejected_logps_token.float())
        token_loss = F.softplus(token_gap)
        weighted_topo_loss = (weighted_topo_mask * token_loss).sum(-1)
        topo_aux_loss = weighted_topo_loss / active_weight_sum.clamp_min(1e-8)
        topo_aux_loss = torch.where(active_weight_sum > 0, topo_aux_loss, torch.zeros_like(topo_aux_loss))
        return topo_aux_loss, weighted_topo_mask, active_weight_sum, active_token_count, high_risk_token_count

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        if "prompt_attention_mask" in batch:
            prompt_lens = batch["prompt_attention_mask"].sum(dim=1)
        else:
            prompt_lens = torch.tensor(
                [0] * batch["chosen_input_ids"].shape[0],
                device=self.accelerator.device,
            )

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

        concatenated_input_ids = torch.cat([c_ids, r_ids], dim=0)
        concatenated_attention_mask = torch.cat([c_mask, r_mask], dim=0)
        concatenated_labels = torch.cat([c_labels, r_labels], dim=0)
        len_chosen = c_ids.shape[0]

        def get_logprobs(logits, labels):
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
            loss_mask = labels != -100
            log_probs = F.log_softmax(logits, dim=-1)
            labels[labels == -100] = 0
            per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
            return per_token_logps, loss_mask

        model_kwargs = {}
        with torch.no_grad():
            if self.ref_model is None:
                unwrapped_model = model.module if hasattr(model, "module") else model
                with unwrapped_model.disable_adapter():
                    ref_output = unwrapped_model(
                        input_ids=concatenated_input_ids,
                        attention_mask=concatenated_attention_mask,
                        use_cache=False,
                        **model_kwargs,
                    )
            else:
                ref_output = self.ref_model(
                    input_ids=concatenated_input_ids,
                    attention_mask=concatenated_attention_mask,
                    use_cache=False,
                    **model_kwargs,
                )
            all_ref_logps, _ = get_logprobs(ref_output.logits, concatenated_labels)
            all_ref_logps = all_ref_logps.detach()

        model_output = model(
            input_ids=concatenated_input_ids,
            attention_mask=concatenated_attention_mask,
            use_cache=False,
            **model_kwargs,
        )
        all_logps, all_masks = get_logprobs(model_output.logits, concatenated_labels)

        chosen_logps_token = all_logps[:len_chosen]
        rejected_logps_token = all_logps[len_chosen:]
        chosen_mask = all_masks[:len_chosen]
        rejected_mask = all_masks[len_chosen:]
        ref_chosen_logps_token = all_ref_logps[:len_chosen]
        ref_rejected_logps_token = all_ref_logps[len_chosen:]

        aligned_topo_mask = torch.zeros_like(rejected_logps_token, dtype=torch.float32)
        if "topo_mask" in batch:
            topo_mask = batch["topo_mask"].to(self.accelerator.device)
            aligned_topo_mask = self._align_topo_mask(topo_mask, prompt_lens, rejected_logps_token.shape)
            metrics[f"{prefix}topo/mask_mean"] = aligned_topo_mask.mean().detach().cpu()

        chosen_logps = (chosen_logps_token * chosen_mask).sum(-1)
        rejected_logps = (rejected_logps_token * rejected_mask).sum(-1)
        ref_chosen_logps = (ref_chosen_logps_token * chosen_mask).sum(-1)
        ref_rejected_logps = (ref_rejected_logps_token * rejected_mask).sum(-1)

        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = (pi_logratios - ref_logratios).float()

        if "margin" in batch:
            raw_margin = batch["margin"].to(self.accelerator.device, dtype=logits.dtype)
            margin = raw_margin * self.margin_scale_train
            metrics[f"{prefix}margin/raw_mean"] = raw_margin.mean().detach().cpu()
            metrics[f"{prefix}margin/scaled_mean"] = margin.mean().detach().cpu()
        else:
            margin = torch.zeros_like(logits)
            zero = torch.zeros((), device=self.accelerator.device).cpu()
            metrics[f"{prefix}margin/raw_mean"] = zero
            metrics[f"{prefix}margin/scaled_mean"] = zero

        main_loss = -F.logsigmoid(self.beta * logits - margin)
        topo_aux_loss = torch.zeros_like(main_loss)
        active_topo_mask = torch.zeros_like(rejected_logps_token, dtype=torch.float32)
        active_weight_sum = torch.zeros_like(main_loss)
        active_token_count = torch.zeros_like(main_loss)
        high_risk_token_count = torch.zeros_like(main_loss)
        if "topo_mask" in batch:
            topo_aux_loss, active_topo_mask, active_weight_sum, active_token_count, high_risk_token_count = self._compute_topo_aux_loss(
                aligned_topo_mask=aligned_topo_mask,
                rejected_mask=rejected_mask,
                rejected_logps_token=rejected_logps_token,
                ref_rejected_logps_token=ref_rejected_logps_token,
            )
            raw_weighted_nll = (
                active_topo_mask
                * F.softplus(self.beta * (rejected_logps_token.float() - ref_rejected_logps_token.float()))
            ).sum(
                -1
            )
            metrics[f"{prefix}valid_error_tokens"] = high_risk_token_count.mean().detach().cpu()
            metrics[f"{prefix}topo/active_tokens"] = active_token_count.mean().detach().cpu()
            metrics[f"{prefix}topo/high_risk_tokens"] = high_risk_token_count.mean().detach().cpu()
            metrics[f"{prefix}topo/active_weight_sum"] = active_weight_sum.mean().detach().cpu()
            metrics[f"{prefix}topo/raw_weighted_nll"] = raw_weighted_nll.mean().detach().cpu()

        total_loss = main_loss + self.lambda_coeff * topo_aux_loss
        loss = total_loss.mean()
        metrics[f"{prefix}main_loss"] = main_loss.mean().detach().cpu()
        metrics[f"{prefix}topo/aux_loss"] = topo_aux_loss.mean().detach().cpu()
        metrics[f"{prefix}total_loss"] = loss.detach().cpu()

        with torch.no_grad():
            chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
            rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()

        return loss, metrics


PEFT_CONFIG = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj"],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-8 training entry for DPO / marginDPO / FaDPO.")
    parser.add_argument("--base_model", type=str, default="/home/models")
    parser.add_argument(
        "--train_dataset",
        type=str,
        default=str(artifact_path("fa_dpo_pipeline", "topodpo_data", "train_fadpo.jsonl")),
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=str(artifact_path("fa_dpo_pipeline", "topodpo_data", "eval_fadpo.jsonl")),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(checkpoint_path("fa_dpo_pipeline", "fadpo")),
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        choices=["dpo", "margindpo", "fadpo", "topodpo"],
        default="fadpo",
        help="dpo=plain baseline, margindpo=margin-only, fadpo=topo-mask+margin, topodpo=compat alias for fadpo",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Preference sharpness / DPO temperature.",
    )
    parser.add_argument(
        "--lambda_coeff",
        type=float,
        default=0.05,
        help="Weight for the token-level topo auxiliary loss.",
    )
    parser.add_argument(
        "--margin_scale_train",
        type=float,
        default=0.2,
        help="Training-side multiplier applied to raw sample margins.",
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument(
        "--metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "08_train_topodpo.json"),
        help="Where to write run metadata JSON.",
    )
    return parser.parse_args()


def canonicalize_mode(mode: str) -> str:
    return "fadpo" if mode == "topodpo" else mode


def apply_mode_defaults(args: argparse.Namespace) -> None:
    args.training_mode = canonicalize_mode(args.training_mode)
    default_train = str(artifact_path("fa_dpo_pipeline", "topodpo_data", "train_fadpo.jsonl"))
    default_eval = str(artifact_path("fa_dpo_pipeline", "topodpo_data", "eval_fadpo.jsonl"))
    default_output = str(checkpoint_path("fa_dpo_pipeline", "fadpo"))
    if args.training_mode == "dpo":
        if args.train_dataset == default_train:
            args.train_dataset = str(artifact_path("fa_dpo_pipeline", "topodpo_data", "train_dpo.jsonl"))
        if args.eval_dataset == default_eval:
            args.eval_dataset = str(artifact_path("fa_dpo_pipeline", "topodpo_data", "eval_dpo.jsonl"))
        if args.output_dir == default_output:
            args.output_dir = str(checkpoint_path("fa_dpo_pipeline", "dpo"))
    elif args.training_mode == "margindpo":
        if args.train_dataset == default_train:
            args.train_dataset = str(artifact_path("fa_dpo_pipeline", "topodpo_data", "train_margindpo.jsonl"))
        if args.eval_dataset == default_eval:
            args.eval_dataset = str(artifact_path("fa_dpo_pipeline", "topodpo_data", "eval_margindpo.jsonl"))
        if args.output_dir == default_output:
            args.output_dir = str(checkpoint_path("fa_dpo_pipeline", "margindpo"))


def train(args: argparse.Namespace) -> tuple[Accelerator, Path]:
    accelerator = Accelerator()
    if accelerator.is_main_process:
        print("\n" + "=" * 50)
        print("--- Preference Training Start ---")
        print(f"  Mode: {args.training_mode}")
        print(f"  Base Model: {args.base_model}")
        print("=" * 50 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=model_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = get_peft_model(model, PEFT_CONFIG)
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    train_dataset = load_dataset("json", data_files=args.train_dataset, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_dataset, split="train")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        logging_steps=5,
        eval_steps=20,
        save_strategy="steps",
        warmup_ratio=0.1,
        save_steps=20,
        bf16=args.bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="tensorboard",
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        truncation_mode="keep_end",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=args.beta,
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs
        )

    model.enable_input_require_grads()
    model.config.use_cache = False

    use_topo_mask = args.training_mode == "fadpo"
    use_margin = args.training_mode in {"margindpo", "fadpo"}
    data_collator = TopoDPODataCollator(
        pad_token_id=tokenizer.pad_token_id,
        use_topo_mask=use_topo_mask,
        use_margin=use_margin,
    )
    trainer_cls = DPOTrainer if args.training_mode == "dpo" else TopoDPOTrainer
    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    if trainer_cls is DPOTrainer:
        dpo_trainer = trainer_cls(**trainer_kwargs)
    else:
        dpo_trainer = trainer_cls(
            **trainer_kwargs,
            lambda_coeff=args.lambda_coeff,
            margin_scale_train=args.margin_scale_train,
        )

    if accelerator.is_main_process:
        print(f"Starting {args.training_mode} training...")

    dpo_trainer.train()
    accelerator.wait_for_everyone()

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        print("Training complete. Saving final model...")
        dpo_trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("--- 脚本执行完毕 ---")
    return accelerator, output_dir


def main() -> None:
    args = parse_args()
    apply_mode_defaults(args)
    started_at = iso_utc_now()

    accelerator: Accelerator | None = None
    output_dir = Path(args.output_dir)
    try:
        accelerator, output_dir = train(args)
    except Exception:
        if accelerator is None or accelerator.is_main_process:
            write_run_metadata(
                stage_name="08_train_topodpo",
                args=args,
                inputs={
                    "train_dataset": args.train_dataset,
                    "eval_dataset": args.eval_dataset,
                },
                outputs={
                    "output_dir": output_dir,
                    "trainer_state": output_dir / "trainer_state.json",
                },
                stats={"trainer_state_exists": (output_dir / "trainer_state.json").exists()},
                metadata_file=args.metadata_file,
                started_at=started_at,
                finished_at=iso_utc_now(),
                status="failed",
            )
        raise

    if accelerator.is_main_process:
        metadata_path = write_run_metadata(
            stage_name="08_train_topodpo",
            args=args,
            inputs={
                "train_dataset": args.train_dataset,
                "eval_dataset": args.eval_dataset,
            },
            outputs={
                "output_dir": output_dir,
                "trainer_state": output_dir / "trainer_state.json",
            },
            stats={"trainer_state_exists": (output_dir / "trainer_state.json").exists()},
            metadata_file=args.metadata_file,
            started_at=started_at,
            finished_at=iso_utc_now(),
        )
        print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
