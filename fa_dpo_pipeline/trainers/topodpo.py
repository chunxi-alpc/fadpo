import argparse
import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal, Dict, List, Union, Optional, Any, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence

# ==========================================
# 1. 自定义 Data Collator
# ==========================================
@dataclass
class TopoDPODataCollator(DataCollatorForPreference):
    """
    继承自 DataCollatorForPreference。
    处理 topo_mask 和 margin 的提取与对齐。
    """
    use_topo_mask: bool = True
    use_margin: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. 提取自定义字段 (Pop出，保持features纯净)
        topo_masks = [f.pop("topo_mask") for f in features] if "topo_mask" in features[0] else None
        margins = [f.pop("margin") for f in features] if "margin" in features[0] else None
        if not self.use_topo_mask:
            topo_masks = None
        if not self.use_margin:
            margins = None
        
        # 2. 调用父类处理常规字段
        batch = super().__call__(features)
        
        # 3. 处理 Topo Mask
        if topo_masks is not None:
            # 转 Tensor
            topo_masks_tensors = [torch.tensor(m, dtype=torch.float32) for m in topo_masks]
            
            # Padding: 使用 0.0 填充
            batch["topo_mask"] = pad_sequence(topo_masks_tensors, batch_first=True, padding_value=0.0)
            
            # === 关键：长度对齐 ===
            # batch["rejected_input_ids"] 可能包含 prompt 和 padding
            # 这里的对齐策略需根据 dataset 中 topo_mask 的生成逻辑定。
            # 假设 topo_mask 是针对 completion 部分的，我们需要将其 Pad 到与 rejected_input_ids 一致
            # (注意：通常父类处理后，rejected_input_ids 是 right-padded)
            rej_len = batch["rejected_input_ids"].shape[1]
            mask_len = batch["topo_mask"].shape[1]
            
            if mask_len < rej_len:
                pad_amt = rej_len - mask_len
                batch["topo_mask"] = F.pad(batch["topo_mask"], (0, pad_amt), value=0.0)
            elif mask_len > rej_len:
                batch["topo_mask"] = batch["topo_mask"][:, :rej_len]

        # 4. 处理 Margin
        if margins is not None:
            batch["margin"] = torch.tensor(margins, dtype=torch.float32)
            
        return batch

class TopoDPOTrainer(DPOTrainer):
    def __init__(self, model, ref_model=None, args=None, lambda_coeff=0.01, **kwargs):
        # 修复 DDP bug
        if args is not None and args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                args.gradient_checkpointing_kwargs = {}
            if args.gradient_checkpointing_kwargs.get("use_reentrant", True):
                if Accelerator().is_local_main_process:
                    print("TopoDPOTrainer: Forcing use_reentrant=False to fix DDP error.")
                args.gradient_checkpointing_kwargs["use_reentrant"] = False
        
        # 保存加法惩罚系数
        self.lambda_coeff = lambda_coeff
        
        super().__init__(model, ref_model=ref_model, args=args, **kwargs)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        # --- 1. 动态 Prompt 长度计算 ---
        if "prompt_attention_mask" in batch:
            prompt_lens = batch["prompt_attention_mask"].sum(dim=1)
        else:
            prompt_lens = torch.tensor([0] * batch["chosen_input_ids"].shape[0], device=self.accelerator.device)

        # --- 2. 构建 Labels (Mask Prompt) ---
        def build_labels_per_sample(input_ids, attention_mask, p_lens):
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            for i, p_len in enumerate(p_lens):
                labels[i, :p_len] = -100
            return labels

        if "chosen_labels" not in batch:
            batch["chosen_labels"] = build_labels_per_sample(
                batch["chosen_input_ids"], batch["chosen_attention_mask"], prompt_lens
            )
        if "rejected_labels" not in batch:
            batch["rejected_labels"] = build_labels_per_sample(
                batch["rejected_input_ids"], batch["rejected_attention_mask"], prompt_lens
            )

        # --- 3. 拼接 Input ---
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
            if tensor.shape[1] >= length: return tensor
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

        # --- 4. Forward Pass ---
        def get_logprobs(logits, labels):
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
            loss_mask = (labels != -100)
            log_probs = F.log_softmax(logits, dim=-1)
            labels[labels == -100] = 0 
            per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
            return per_token_logps, loss_mask

        model_kwargs = {}
        
        # Ref Model
        with torch.no_grad():
            if self.ref_model is None:
                if hasattr(model, "module"): unwrapped_model = model.module
                else: unwrapped_model = model
                with unwrapped_model.disable_adapter():
                    ref_output = unwrapped_model(input_ids=concatenated_input_ids, attention_mask=concatenated_attention_mask, use_cache=False, **model_kwargs)
            else:
                ref_output = self.ref_model(input_ids=concatenated_input_ids, attention_mask=concatenated_attention_mask, use_cache=False, **model_kwargs)
            all_ref_logps, _ = get_logprobs(ref_output.logits, concatenated_labels)
            all_ref_logps = all_ref_logps.detach()

        # Policy Model
        model_output = model(input_ids=concatenated_input_ids, attention_mask=concatenated_attention_mask, use_cache=False, **model_kwargs)
        all_logps, all_masks = get_logprobs(model_output.logits, concatenated_labels)

        # --- 5. Logprobs 切分与 Mask 准备 ---
        chosen_logps_token = all_logps[:len_chosen]
        rejected_logps_token = all_logps[len_chosen:]
        chosen_mask = all_masks[:len_chosen]
        rejected_mask = all_masks[len_chosen:] 
        ref_chosen_logps_token = all_ref_logps[:len_chosen]
        ref_rejected_logps_token = all_ref_logps[len_chosen:]

        # --- 6. Topo Mask 对齐逻辑 ---
        # 注意：这里我们只计算对齐后的 mask，不把它乘进 logprobs 里！
        aligned_topo_mask = torch.zeros_like(rejected_logps_token, dtype=torch.float32)
        
        if "topo_mask" in batch:
            topo_mask = batch["topo_mask"].to(self.accelerator.device)
            for i in range(len_chosen):
                cur_p_len = prompt_lens[i].item()
                start_idx = max(0, cur_p_len - 1)
                raw_mask_content = topo_mask[i]
                fill_len = min(raw_mask_content.shape[0], aligned_topo_mask.shape[1] - start_idx)
                if fill_len > 0:
                    aligned_topo_mask[i, start_idx : start_idx + fill_len] = raw_mask_content[:fill_len]
            
            metrics["topo_mask_mean"] = aligned_topo_mask.mean().detach().cpu()

        # ==========================================================
        # 7. Loss 计算 (修正后的 Additive Penalty)
        # ==========================================================
        
        # A. 标准 Logprobs 求和 (不加权，保证数值稳定性)
        chosen_logps = (chosen_logps_token * chosen_mask).sum(-1)
        rejected_logps = (rejected_logps_token * rejected_mask).sum(-1)
        ref_chosen_logps = (ref_chosen_logps_token * chosen_mask).sum(-1)
        ref_rejected_logps = (ref_rejected_logps_token * rejected_mask).sum(-1)

        # B. 标准 LogRatios
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios

        # C. 计算 Topo Penalty (关键修正！)
        # 逻辑：Dataset 中，正确区域是 EPSILON (0.1)，错误区域是 > 1.0 的值。
        # 我们只希望惩罚错误区域带来的"额外风险"。
        # 如果直接 sum，长句子会有巨大的 penalty。
        if "topo_mask" in batch:
            # 1. 找出那些真正是错误的区域 (值 > 1.0 的部分)
            #    假设 dataset 里的 weight = 1 + alpha * score
            #    我们设定一个基准线 1.0 (代表无惩罚的标准权重)
            #    或者更简单：只计算 mask > 0.5 (即非 epsilon) 的部分
            
            # 方法：创建一个只包含"错误惩罚"的 mask
            # 如果 aligned_topo_mask[i] < 1.0 (是epsilon)，则视为 0 惩罚
            # 如果 aligned_topo_mask[i] > 1.0，则惩罚值为 (mask_val) 本身 (或者 mask_val - 1.0)
            
            # 建议策略：为了保留你 dataset 中 alpha 的物理意义，直接取值即可，
            # 但必须把 epsilon 过滤掉！
            
            # 过滤掉 epsilon (正确区域)
            error_regions = torch.where(
                aligned_topo_mask > 0.5,  # 阈值设为 0.5，过滤掉 0.1 的 epsilon
                aligned_topo_mask,        # 如果是错误，保留权重
                torch.tensor(0.0, device=self.accelerator.device) # 如果是正确，惩罚为 0
            )
            
            # 计算总惩罚值
            valid_topo_penalty = (error_regions * rejected_mask).sum(-1)
            
            # 应用惩罚
            # 这里的 lambda_coeff 控制"每一个错误token"对 Margin 的影响
            penalty_term = self.lambda_coeff * valid_topo_penalty
            
            # Logits 减去惩罚项 (Logits 变小 -> Loss 变大 -> 强迫模型优化)
            logits = logits - penalty_term
            
            metrics["avg_penalty_val"] = penalty_term.mean().detach().cpu()
            metrics["valid_error_tokens"] = (error_regions > 0).float().sum(-1).mean().cpu()

        # D. 处理 Global Margin (来自 Dataset 的 max_risk)
        if "margin" in batch:
            margin = batch["margin"].to(self.accelerator.device)
            # 此时的 logits 已经包含了 token-level 的 penalty
            # 再减去 sequence-level 的 margin
            loss = -F.logsigmoid(self.beta * logits - margin).mean()
        else:
            loss = -F.logsigmoid(self.beta * logits).mean()

        # Metrics (记录供观察)
        with torch.no_grad():
            chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
            rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        
        return loss, metrics
    
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'] 
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--training_mode",
        type=str,
        choices=["dpo", "fadpo", "topodpo"],
        default="topodpo",
        help="dpo=plain baseline, fadpo=margin-only, topodpo=margin+topo-mask",
    )
    
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true", default=True)
    # parser.add_argument("--resume_from_checkpoint", type=str, default="checkpoints/UltraMedical2_curriculum/checkpoint-180")
    args = parser.parse_args()

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("--- Preference Training Start ---")
        print(f"  Mode: {args.training_mode}")
        print(f"  Base Model: {args.base_model}")
        print("="*50 + "\n")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # 很多模型默认 pad_token_id 是 None，设为 eos
        # 注意：下面的 DataCollator 和 Trainer 已经处理了 Pad==EOS 时的 Label Masking 问题
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation='flash_attention_2'
    )
    model = get_peft_model(model, peft_config)
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # Load Datasets
    train_dataset = load_dataset("json", data_files=args.train_dataset, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_dataset, split="train")
    
    # Shuffle
    # train_dataset = train_dataset.shuffle(seed=42)

    # Config
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
        remove_unused_columns=False, # 必须 False
        report_to="tensorboard",
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        truncation_mode="keep_end",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # precompute_ref_log_probs=True, 
        beta=0.3,
    )
    
    # 手动启用 Gradient Checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs
        )
    
    model.enable_input_require_grads()
    model.config.use_cache = False
    
    use_topo_mask = args.training_mode == "topodpo"
    use_margin = args.training_mode in {"fadpo", "topodpo"}
    data_collator = TopoDPODataCollator(
        pad_token_id=tokenizer.pad_token_id,
        use_topo_mask=use_topo_mask,
        use_margin=use_margin,
    )
    trainer_cls = DPOTrainer if args.training_mode == "dpo" else TopoDPOTrainer

    # Initialize trainer
    dpo_trainer = trainer_cls(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    if accelerator.is_main_process:
        print(f"Starting {args.training_mode} training...")
        
    dpo_trainer.train()
    # dpo_trainer.train(resume_from_checkpoint=args.resume_from_checkpoint) 
    
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("Training complete. Saving final model...")
        dpo_trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if accelerator.is_main_process:
        print("--- 脚本执行完毕 ---")
