"""Response-only supervised fine-tuning for the P1 decoder."""
import json
import math
import yaml
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer
from ..decoding.sequence_generator import SequenceGenerator
from ..utils.webnlg import format_webnlg_prompt, score_webnlg


class SFTTrainer(BaseTrainer):
    """Train a decoder on prompt/response pairs.

    Complete the marked training, validation, and generation methods.
    Reuse your P1 model and decoding implementation.
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        self.best_metric = -float("inf")  # Higher Kaggle score is better.
        self.amp_enabled = bool(config["training"].get("amp", True)) and torch.device(self.device).type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        # TODO: Use summed cross-entropy, ignoring labels equal to -100.
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")  # TODO

    def _init_experiment(self, run_name, config_file):
        """Provided: use a new output directory, or explicitly resume an existing run."""
        from time import time_ns
        root = (
            Path(self.config.get("output_dir", "expts"))
            .expanduser()
            .resolve()
            / run_name
        )
        resume = self.config["training"].get("resume", False)
        if resume:
            last_checkpoint = root / "checkpoints" / "last.pth"
            if not last_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Cannot resume: {last_checkpoint} was not found. "
                    "If training stopped before the first checkpoint, set this run's resume flag to false and rerun."
                )
        else:
            if root.exists():
                # Preserve any experiment containing saved model weights.
                has_checkpoint = any(
                    path.is_file()
                    and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
                    for path in root.rglob("*")
                )
                if has_checkpoint:
                    raise FileExistsError(
                        f"Experiment {run_name!r} already contains saved model weights. "
                        "Set resume to true to continue from last.pth, or choose a new run name."
                    )
                # Finish the notebook's previous W&B session before
                # moving its files.
                if self.use_wandb:
                    import wandb
                    if wandb.run is not None:
                        wandb.finish()
                archived_root = root.with_name(
                    f"{root.name}-before-first-checkpoint-{time_ns()}"
                )
                root.rename(archived_root)
                print(
                    "Note: The previous attempt has no saved checkpoint. "
                    f"Its files were preserved at {archived_root}. "
                    "Starting this experiment again from the P1 checkpoint."
                )
            root.mkdir(parents=True, exist_ok=False)
            (root / "config.yaml").write_text(yaml.safe_dump(self.config))
            (root / "model.txt").write_text(str(self.model) + "\n")
        checkpoint, attn, text = (root / part for part in ("checkpoints", "attn", "text"))
        for directory in (checkpoint, attn, text):
            directory.mkdir(exist_ok=True)
        if self.use_wandb:
            import wandb
            if wandb.run is not None:
                wandb.finish()
            self.wandb_run = wandb.init(
                project=self.config["training"].get("wandb_project", "hw4p2"),
                entity=self.config["training"].get("wandb_entity") or None,
                name=run_name, config=self.config,
                id=self.config["training"].get("wandb_run_id") or None,
                resume="must" if resume else None, dir=str(root))
        return root, checkpoint, attn, text, checkpoint / "best.pth", checkpoint / "last.pth"


    @staticmethod
    def _metrics(loss_sum, token_count):
        """Provided: perplexity here is response-token PPL, not character PPL."""
        if token_count <= 0:
            raise ValueError("No supervised response tokens")
        ce = loss_sum / token_count
        if not math.isfinite(ce):
            raise FloatingPointError("Non-finite response loss")
        return {"response_ce": ce, "response_ppl": math.exp(ce), "response_tokens": token_count}

    def _batch(self, batch):
        """Provided: move tensors and reject a batch without supervised targets."""
        inputs, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)
        if inputs.ndim != 2 or inputs.shape != labels.shape:
            raise ValueError("input_ids and labels must have matching (B, T) shapes")
        count = int((labels != -100).sum())
        if count == 0:
            raise ValueError("Batch has no supervised response tokens")
        return inputs, labels, count

    def _train_epoch(self, dataloader):
        """Accumulate summed losses; normalize gradients by the cycle's token count.

        Handle a partial final accumulation cycle. Unscale before clipping.
        Advance the scheduler only when the optimizer step ran.
        """
        # Provided: keep this call when implementing the training loop.
        self._bind_model_to_active_wandb()
        # TODO: Implement one training epoch using the instructions above.

        num_batches = len(dataloader)
        if num_batches == 0:
            raise ValueError("Training dataloader must contain at least one batch")
        accumulation_steps = self.config["training"].get("gradient_accumulation_steps", 1)
        if not isinstance(accumulation_steps, int) or isinstance(accumulation_steps, bool) or accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
        device_type = torch.device(self.device).type
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
 
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, token_count, cycle_tokens = 0.0, 0, 0
        batch_bar = tqdm(total=num_batches, dynamic_ncols=True, leave=False, desc="[Training SFT]")
 
        for batch_index, batch in enumerate(dataloader):
            inputs, labels, count = self._batch(batch)
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=self.amp_enabled):
                logits, _ = self.model(inputs)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1))
            self.scaler.scale(loss).backward()
 
            loss_sum += loss.item()
            token_count += count
            cycle_tokens += count
 
            should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == num_batches
            if should_step:
                self.scaler.unscale_(self.optimizer)
                for parameter in trainable_parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(cycle_tokens)
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
                scale_before_step = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                step_ran = self.scaler.get_scale() >= scale_before_step
                if step_ran:
                    self.global_optimizer_step = getattr(self, "global_optimizer_step", 0) + 1
                    if self.scheduler is not None and not isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                cycle_tokens = 0
 
            batch_bar.set_postfix(response_ce=f"{loss_sum / token_count:.4f}")
            batch_bar.update()
 
        batch_bar.close()
        return self._metrics(loss_sum, token_count)

    @torch.inference_mode()
    def _validate_epoch(self, dataloader):
        """Compute response CE from total NLL / total supervised tokens."""
        # TODO: Evaluate without gradients; exclude prompt and padding targets.
        if len(dataloader) == 0:
            raise ValueError("Validation dataloader must contain at least one batch")
        self.model.eval()
        loss_sum, token_count = 0.0, 0
        for batch in tqdm(dataloader, dynamic_ncols=True, leave=False, desc="[Validating SFT]"):
            inputs, labels, count = self._batch(batch)
            logits, _ = self.model(inputs)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1))
            loss_sum += loss.item()
            token_count += count
        return self._metrics(loss_sum, token_count)

    @torch.inference_mode()
    def generate(self, records):
        """Generate from the triples alone, using your P1 greedy decoder.

        Do not append references to prompts. Decode one unpadded prompt at a time.
        Limit total context and stop at generated EOS, excluding prompt text.
        """
        # TODO: Use SequenceGenerator and the model's score/score_cached methods.
        self.model.eval()
        max_context = self.model.max_len
        max_new_tokens = self.config.get("generation", {}).get("max_new_tokens")
        use_cache = bool(getattr(self.model, "use_rope", False))
        predictions = []
        for record in tqdm(records, dynamic_ncols=True, leave=False, desc="[Generating SFT]"):
            prompt_ids = self.tokenizer.encode(format_webnlg_prompt(record["triples"]))
            if not prompt_ids or len(prompt_ids) >= max_context:
                raise ValueError(f"Prompt for {record.get('id')} does not fit the model context")
            max_length = max_context
            if max_new_tokens is not None:
                max_length = min(max_context, len(prompt_ids) + int(max_new_tokens))
            generator = SequenceGenerator(
                score_fn=self.model.score,
                tokenizer=self.tokenizer,
                max_length=max_length,
                device=self.device,
                cache_score_fn=self.model.score_cached if use_cache else None,
            )
            prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            sequences, _ = generator.generate_greedy(prompt)
            generated_ids = sequences[0, len(prompt_ids):].tolist()
            if self.tokenizer.eos_id in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(self.tokenizer.eos_id)]
            predictions.append(self.tokenizer.decode(generated_ids).strip())
        return predictions

    def evaluate(self, dataloader, records=None):
        """Provided: combine teacher-forced response loss with generated-text metrics."""
        metrics = self._validate_epoch(dataloader)
        if records is not None:
            predictions = self.generate(records)
            scores = score_webnlg(predictions, [row["references"] for row in records])
            metrics.update(bleu=scores["bleu"], rouge_l=scores["rouge_l"],
                           kaggle_score=scores["kaggle_score"], examples=len(records))
        return metrics

    def train(self, train_dataloader, val_dataloader, dev_records):
        """Run epochs and select the best generated dev Kaggle-score checkpoint.

        Save last.pth for resuming the next epoch and best.pth for prediction.
        Log token loss and generation scores as different quantities.
        """
        # TODO: Run training/evaluation, update best Kaggle score, log, and save checkpoints.
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first")
        total_epochs = self.config["training"]["epochs"]
        for epoch in range(self.current_epoch, total_epochs):
            train_metrics = self._train_epoch(train_dataloader)
            val_metrics = self.evaluate(val_dataloader, dev_records)
 
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["response_ce"])
 
            token_metrics = {key: val_metrics[key] for key in ("response_ce", "response_ppl", "response_tokens")}
            generation_metrics = {key: val_metrics[key] for key in ("bleu", "rouge_l", "kaggle_score", "examples")}
            self._log_metrics(
                {"train": train_metrics, "val": token_metrics, "val_generation": generation_metrics},
                epoch,
            )
 
            is_best = val_metrics["kaggle_score"] > self.best_metric
            if is_best:
                self.best_metric = val_metrics["kaggle_score"]
 
            self.current_epoch = epoch + 1
            self.save_checkpoint("last.pth")
            if is_best:
                self.save_checkpoint("best.pth")
 
