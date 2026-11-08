"""Training, validation, and generation for the HW4P1 language model."""

from .base_trainer import BaseTrainer
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Optional
from torch.utils.data import DataLoader
from ..data import H4Tokenizer
from ..decoding.sequence_generator import SequenceGenerator


MetricDict = dict[str, float]
AttentionDict = dict[str, torch.Tensor]
GenerationResult = dict[str, Any]


class LMTrainer(BaseTrainer):
    """Language Model Trainer class that handles training, validation, and generation.

    This trainer implements:
    1. Training with gradient accumulation and mixed precision
    2. Validation for model evaluation
    3. Generation with greedy and sampling strategies (beam search is optional)
    4. Metric, attention, generated-text, and checkpoint logging

    TODO:
    - Initialize summed next-token cross-entropy in ``__init__``.
    - Complete the fixed-chunk training and validation calculations.
    - Handle full and partial gradient-accumulation cycles correctly.
    - Connect the epoch loop to training, validation, and generation.
    - Call your greedy and sampling implementations from ``generate``.

    Implementation Notes:
    1. GPT-2 shards already contain shifted fixed-length windows, so no padding
       index or per-example length tensor is needed.
    2. Cross-entropy uses ``reduction='sum'`` so gradients can be normalized by
       the actual token count in each accumulation cycle.
    3. Cached generation is used only for rotary decoder models.
    4. Character perplexity uses the corpus statistic stored in the shard
       manifest.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: H4Tokenizer,
        config: dict,
        run_name: str,
        config_file: str,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the language-model trainer.

        Args:
            model: Decoder-only language model.
            tokenizer: Tokenizer used to build the token shards.
            config: Training configuration.
            run_name: Experiment directory name.
            config_file: Configuration file copied into the experiment directory.
            device: Optional PyTorch device string.

        """

        super().__init__(model, tokenizer, config, run_name, config_file, device)
        # TODO: Initialize nn.CrossEntropyLoss. Use label_smoothing from the
        # training config and reduction="sum" so each accumulation cycle can
        # be normalized by its exact token count.
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config["training"].get("label_smoothing", 0.0),
            reduction="sum",
        )


    @staticmethod
    def _optimizer_step_ran(
        scale_before_step: float,
        scale_after_step: float,
    ) -> bool:
        """Check whether GradScaler skipped an optimizer update.

        Args:
            scale_before_step: Loss scale before ``GradScaler.update``.
            scale_after_step: Loss scale after ``GradScaler.update``.

        Returns:
            ``False`` when an overflow reduced the scale and skipped the step.
        """
        step_ran = scale_after_step >= scale_before_step
        return step_ran

    @staticmethod
    def _avg_chars_per_token(dataloader: DataLoader) -> float:
        """Read the corpus characters-per-token ratio.

        Args:
            dataloader: DataLoader whose dataset stores corpus statistics.

        Returns:
            Corpus characters divided by serialized tokens.

        """
        value = float(dataloader.dataset.get_avg_chars_per_token())
        return value

    @staticmethod
    def _attention_snapshot(
        attention_weights: AttentionDict,
    ) -> Optional[AttentionDict]:
        """Detach one attention map for plotting.

        Args:
            attention_weights: Per-layer attention tensors for one batch.

        Returns:
            One single-example attention tensor, or ``None`` when unavailable.
        """
        if not attention_weights:
            snapshot = None
        else:
            first_key = next(iter(attention_weights))
            first_attention = attention_weights[first_key][:1].detach().cpu()
            snapshot = {first_key: first_attention}
        return snapshot

    def _train_epoch(
        self,
        dataloader: DataLoader,
    ) -> tuple[MetricDict, AttentionDict]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader returning fixed input/target chunks.

        Returns:
            Token/character metrics and one attention snapshot.

        Raises:
            ValueError: If the DataLoader or accumulation setting is invalid.
        """
        num_batches = len(dataloader)
        if num_batches == 0:
            raise ValueError("Training dataloader must contain at least one batch")

        accumulation_steps = self.config["training"].get(
            "gradient_accumulation_steps",
            1,
        )
        if (
            not isinstance(accumulation_steps, int)
            or isinstance(accumulation_steps, bool)
            or accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps must be a positive integer")

        # Initialize training variables
        self.model.train()
        self._bind_model_to_active_wandb()
        batch_bar = tqdm(
            total=num_batches,
            dynamic_ncols=True,
            leave=False,
            position=0,
            desc="[Training LM]",
        )
        running_ce_loss = 0.0
        total_tokens = 0
        cycle_tokens = 0
        attention_snapshot = None
        device_type = torch.device(self.device).type
        use_autocast = device_type == "cuda"

        # Only zero gradients when starting a new accumulation cycle
        self.optimizer.zero_grad()

        for batch_index, batch in enumerate(dataloader):
            # TODO: Unpack the fixed (inputs, targets) batch and move both
            # tensors to self.device.
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=use_autocast,
            ):
                # TODO: Run the model. Flatten logits to (B*T, vocab) and
                # targets to (B*T), then calculate summed cross-entropy.
                raw_preds, attn_weights = self.model(inputs)
                flat_predictions = raw_preds.reshape(-1, raw_preds.size(-1))
                flat_targets = targets.reshape(-1)
                raw_loss = self.criterion(flat_predictions, flat_targets)

            # Calculate metrics with raw loss (DO NOT MODIFY THIS)
            batch_tokens = targets.numel()
            total_tokens += batch_tokens
            cycle_tokens += batch_tokens
            running_ce_loss += raw_loss.item()

            cycle_start = (batch_index // accumulation_steps) * accumulation_steps
            cycle_size = min(accumulation_steps, num_batches - cycle_start)
            # TODO: Scale raw_loss and backpropagate it.
            self.scaler.scale(raw_loss).backward()

            if attention_snapshot is None:
                attention_snapshot = self._attention_snapshot(attn_weights)

            # Only update weights after accumulating enough gradients
            # Handle any remaining gradients at the end of the epoch
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == num_batches
            )
            if should_step:
                # TODO: Unscale the gradients and divide every existing
                # gradient by cycle_tokens. Step and update the scaler, then
                # use _optimizer_step_ran to detect an overflow-skipped step.
                self.scaler.unscale_(self.optimizer)
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(cycle_tokens)
                scale_before_step = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after_step = self.scaler.get_scale()
                optimizer_step_ran = self._optimizer_step_ran(scale_before_step, scale_after_step)

                # DO NOT MODIFY: count only optimizer updates that AMP did not
                # skip because of an overflow.
                if optimizer_step_ran:
                    self.global_optimizer_step += 1
                # Only step scheduler here if it's not ReduceLROnPlateau
                if (
                    optimizer_step_ran
                    and self.scheduler is not None
                    and not isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    )
                ):
                    self.scheduler.step()
                self.optimizer.zero_grad()
                cycle_tokens = 0

            # Calculate metrics
            avg_ce_loss = running_ce_loss / total_tokens
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
            batch_bar.set_postfix(
                ce_loss_token=f"{avg_ce_loss:.4f}",
                perplexity_token=f"{perplexity_token:.4f}",
                acc_step=f"{batch_index - cycle_start + 1}/{cycle_size}",
            )
            batch_bar.update()

        # Compute final metrics
        batch_bar.close()
        avg_ce_loss = running_ce_loss / total_tokens
        avg_ce_loss_char = avg_ce_loss / self._avg_chars_per_token(dataloader)
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss)).item()
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss_char)).item()
        metrics = {
            "ce_loss_token": avg_ce_loss,
            "ce_loss_char": avg_ce_loss_char,
            "perplexity_token": avg_perplexity_token,
            "perplexity_char": avg_perplexity_char,
        }
        if attention_snapshot is None:
            attention_snapshot = {}
        epoch_result = (metrics, attention_snapshot)
        return epoch_result

    def _validate_epoch(
        self,
        dataloader: DataLoader,
    ) -> tuple[MetricDict, AttentionDict]:
        """Evaluate one epoch without gradients.

        Args:
            dataloader: DataLoader returning fixed input/target chunks.

        Returns:
            Token/character metrics and one attention snapshot.

        Raises:
            ValueError: If the DataLoader is empty or incompatible.
        """
        num_batches = len(dataloader)
        if num_batches == 0:
            raise ValueError("Validation dataloader must contain at least one batch")

        # Initialize validation variables
        self.model.eval()
        batch_bar = tqdm(
            total=num_batches,
            dynamic_ncols=True,
            leave=False,
            position=0,
            desc="[Validating LM]",
        )
        running_ce_loss = 0.0
        total_tokens = 0
        attention_snapshot = None

        with torch.inference_mode():
            for batch in dataloader:
                # TODO: Unpack and move the fixed input/target batch, run the
                # model, flatten logits and targets, and calculate summed loss.
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                raw_preds, attn_weights = self.model(inputs)
                flat_predictions = raw_preds.reshape(-1, raw_preds.size(-1))
                flat_targets = targets.reshape(-1)
                loss_sum = self.criterion(flat_predictions, flat_targets)

                if attention_snapshot is None:
                    attention_snapshot = self._attention_snapshot(attn_weights)

                batch_tokens = targets.numel()
                total_tokens += batch_tokens
                running_ce_loss += loss_sum.item()
                # Update the progress bar
                avg_ce_loss = running_ce_loss / total_tokens
                perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
                batch_bar.set_postfix(
                    ce_loss_token=f"{avg_ce_loss:.4f}",
                    perplexity_token=f"{perplexity_token:.4f}",
                )
                batch_bar.update()

        # Compute final metrics
        batch_bar.close()
        avg_ce_loss = running_ce_loss / total_tokens
        avg_ce_loss_char = avg_ce_loss / self._avg_chars_per_token(dataloader)
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss)).item()
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss_char)).item()
        metrics = {
            "ce_loss_token": avg_ce_loss,
            "ce_loss_char": avg_ce_loss_char,
            "perplexity_token": avg_perplexity_token,
            "perplexity_char": avg_perplexity_char,
        }
        if attention_snapshot is None:
            attention_snapshot = {}
        epoch_result = (metrics, attention_snapshot)
        return epoch_result

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
    ) -> None:
        """Run training, validation, generation, and checkpointing.

        Args:
            train_dataloader: Training DataLoader.
            val_dataloader: Validation DataLoader.
            epochs: Number of epochs to run.

        Returns:
            ``None``.

        Raises:
            ValueError: If optimizer, scheduler, or epoch count is invalid.
        """
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized, initialize it first")
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first")
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")

        # Training loop
        best_val_loss = self.best_metric
        stop_epoch = self.current_epoch + epochs
        for epoch in range(self.current_epoch, stop_epoch):
            if hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)
            # TODO: Train for one epoch, validate, and generate from the
            # validation data.
            train_metrics, train_attention = self._train_epoch(train_dataloader)
            val_metrics, val_attention = self._validate_epoch(val_dataloader)
            gen_results = self.generate(val_dataloader)

            # Step ReduceLROnPlateau scheduler with validation loss
            if isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.scheduler.step(val_metrics["ce_loss_char"])

            # Log metrics
            metrics = {"train": train_metrics, "val": val_metrics}
            self._log_metrics(metrics, epoch)

            # Save attention plots
            if train_attention:
                attention_key = next(iter(train_attention))
                attention = train_attention[attention_key][0]
                self._save_attention_plot(attention, epoch, "train_self")
            if val_attention:
                attention_key = next(iter(val_attention))
                attention = val_attention[attention_key][0]
                self._save_attention_plot(attention, epoch, "val_self")

            # Save generated text
            self._save_generated_text(gen_results, f"val_epoch_{epoch}")
            val_loss = val_metrics["ce_loss_char"]
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.best_metric = val_loss

            # Save checkpoints
            self.current_epoch = epoch + 1
            self.save_checkpoint("checkpoint-last-epoch-model.pth")
            # Check if this is the best model
            if is_best:
                self.save_checkpoint("checkpoint-best-metric-model.pth")

    def evaluate(
        self,
        test_dataloader: DataLoader,
    ) -> tuple[MetricDict, dict[str, list[GenerationResult]]]:
        """Evaluate metrics and every configured decoding strategy.

        Args:
            test_dataloader: Test DataLoader.

        Returns:
            Test metrics and generation results keyed by strategy.

        Raises:
            ValueError: If the data or any generation configuration is invalid.
        """
        test_metrics, test_attention = self._validate_epoch(test_dataloader)
        # Log metrics
        self._log_metrics({"test": test_metrics}, self.current_epoch)

        # Save attention plots
        if test_attention:
            attention_key = next(iter(test_attention))
            attention = test_attention[attention_key][0]
            self._save_attention_plot(attention, self.current_epoch, "test_self")

        # Generate with evaluation configs and collect results
        generation_results = {}
        eval_configs = self._get_evaluation_generation_configs(
            test_dataloader
        )
        for config_name, generation_config in eval_configs.items():
            gen_results = self.generate(
                test_dataloader,
                generation_config=generation_config,
            )
            generation_results[config_name] = gen_results
            suffix = f"test_epoch_{self.current_epoch}_{config_name}"
            self._save_generated_text(gen_results, suffix)

        evaluation = (test_metrics, generation_results)
        return evaluation

    def generate(
        self,
        dataloader: DataLoader,
        generation_config: Optional[dict[str, Any]] = None,
    ) -> list[GenerationResult]:
        """Generate continuations from sampled dataset prompts.

        Args:
            dataloader: DataLoader whose dataset supports ``sample_prompts``.
            generation_config: Optional settings for sample count, prompt and
                maximum lengths, seed, sampling temperature, beam width,
                repetition penalty, top-k, top-p, and cache use.

        Returns:
            Decoded prompts, source continuations, generated continuations,
            and sequence scores.

        Raises:
            ValueError: If generation settings conflict or exceed model limits.
        """

        # Greedy search (default)
        if generation_config is None:
            config = {
                "num_samples": 10,
                "seed": 11785,
                "max_length": self.model.max_len,
                "beam_width": 1,
                "repeat_penalty": 1.0,
                "top_k": 0,
                "top_p": 0.0,
                "use_cache": bool(getattr(self.model, "use_rope", False)),
            }
        else:
            config = generation_config.copy()

        requested_max_length = config.get("max_length", self.model.max_len)
        if (
            not isinstance(requested_max_length, int)
            or isinstance(requested_max_length, bool)
            or requested_max_length <= 0
        ):
            raise ValueError("max_length must be a positive integer")
        effective_max_length = min(requested_max_length, self.model.max_len)

        dataset_context = getattr(dataloader.dataset, "context_length", None)
        default_prompt_limit = max(1, effective_max_length - 1)
        if dataset_context is not None:
            default_prompt_limit = min(default_prompt_limit, dataset_context - 1)
        if default_prompt_limit < 1:
            raise ValueError(
                "Dataset context and model max_len do not leave room for a prompt"
            )

        prompt_length = config.get(
            "prompt_length",
            min(20, default_prompt_limit),
        )
        if (
            not isinstance(prompt_length, int)
            or isinstance(prompt_length, bool)
            or prompt_length <= 0
        ):
            raise ValueError("prompt_length must be a positive integer")
        if prompt_length > effective_max_length:
            raise ValueError("prompt_length must not exceed max_length")
        if dataset_context is not None and prompt_length >= dataset_context:
            raise ValueError(
                "prompt_length must be smaller than the dataset context_length"
            )

        seed = config.get("seed", 11785)
        temperature = config.get("temperature", 1.0)
        repeat_penalty = config.get("repeat_penalty", 1.0)
        beam_width = config.get("beam_width", 1)
        top_k = config.get("top_k", 0)
        top_p = config.get("top_p", 0.0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be greater than zero")
        if not isinstance(beam_width, int) or isinstance(beam_width, bool):
            raise ValueError("beam_width must be an integer")
        if beam_width < 1:
            raise ValueError("beam_width must be at least one")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be between zero and one")

        sampling_enabled = top_k > 0 or top_p > 0
        if sampling_enabled and beam_width > 1:
            raise ValueError("Sampling controls cannot be combined with beam search")
        if sampling_enabled and temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        default_cache_setting = bool(getattr(self.model, "use_rope", False))
        use_cache = config.get("use_cache", default_cache_setting)
        if not isinstance(use_cache, bool):
            raise ValueError("use_cache must be Boolean")
        if use_cache and not getattr(self.model, "use_rope", False):
            raise ValueError("use_cache=True requires a rotary decoder")
        if use_cache:
            cache_score_fn = self.model.score_cached
        else:
            cache_score_fn = None
        # Create sequence generator
        generator = SequenceGenerator(
            score_fn=self.model.score,
            tokenizer=self.tokenizer,
            max_length=effective_max_length,
            device=self.device,
            cache_score_fn=cache_score_fn,
        )

        # Sample prompts and get original sequences
        prompts, originals = dataloader.dataset.sample_prompts(
            num_samples=config.get("num_samples", 10),
            prompt_length=prompt_length,
            seed=seed,
        )
        prompts = prompts.to(self.device)

        # Generate sequences based on method
        self.model.eval()
        with torch.inference_mode():
            if sampling_enabled:
                effective_top_p = top_p if top_p > 0 else 1.0
                # TODO: Call generator.generate_sample with the prompts,
                # temperature, top-k, top-p, repeat penalty, and seed.
                sequences, scores = generator.generate_sample(
                    prompts,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=effective_top_p,
                    repeat_penalty=repeat_penalty,
                    seed=seed,
                )
            elif beam_width > 1:
                # Beam search is not required or graded. If you implement it,
                # call generate_beam here and select beam 0 below.
                beam_sequences, beam_scores = generator.generate_beam(
                    prompts,
                    beam_width=beam_width,
                    repeat_penalty=repeat_penalty,
                )
                # Take best beam
                sequences = beam_sequences[:, 0]
                scores = beam_scores[:, 0]
            else:
                # TODO: Call generator.generate_greedy with the prompts and
                # repeat penalty.
                sequences, scores = generator.generate_greedy(
                    prompts,
                    repeat_penalty=repeat_penalty,
                )

        # Post-process sequences (trim upto EOS token)
        processed_sequences = generator.post_process_sequence(
            sequences,
            self.tokenizer,
            start_index=prompts.size(1),
        )

        # Compile results
        # results is a dictionary with the following keys:
        # - prompt: the decoded prompt
        # - generated: the decoded generated sequence after the prompt
        # - original: the decoded original sequence after the prompt
        # - score: the score of the generated sequence
        # NOTE: You might find the H4Tokenizer class useful here
        results = []
        rows = zip(prompts, processed_sequences, scores, originals)
        for prompt, sequence, score, original in rows:
            processed_original = generator.post_process_sequence(
                original,
                self.tokenizer,
                start_index=len(prompt),
            )
            prompt_ids = prompt.tolist()
            original_ids = processed_original[len(prompt):].tolist()
            generated_ids = sequence[len(prompt):].tolist()
            prompt_text = self.tokenizer.decode(prompt_ids)
            original_text = self.tokenizer.decode(original_ids)
            generated_text = self.tokenizer.decode(generated_ids)
            result = {
                "prompt": prompt_text,
                "original": original_text,
                "generated": generated_text,
                "score": score.item(),
            }
            results.append(result)

        return results

    def _get_evaluation_generation_configs(
        self,
        dataloader: Optional[DataLoader] = None,
    ) -> dict[str, dict[str, Any]]:
        """Build deterministic greedy and sampling settings.

        Args:
            dataloader: Optional DataLoader used to limit prompt length.

        Returns:
            Generation settings keyed by decoding strategy.
        """
        prompt_limit = max(1, self.model.max_len - 1)
        if dataloader is not None:
            dataset_context = getattr(dataloader.dataset, "context_length", None)
            if dataset_context is not None:
                prompt_limit = min(prompt_limit, dataset_context - 1)
        prompt_length = max(1, min(10, prompt_limit))

        generation_length = min(self.model.max_len, prompt_length + 50)
        common_config = {
            "num_samples": 10,
            "prompt_length": prompt_length,
            "seed": 11785,
            "max_length": generation_length,
        }
        greedy_config = common_config.copy()
        greedy_config.update(
            {
                "beam_width": 1,
                "repeat_penalty": 1.0,
                "top_k": 0,
                "top_p": 0.0,
            }
        )
        sample_config = common_config.copy()
        sample_config.update(
            {
                "temperature": 1.0,
                "beam_width": 1,
                "repeat_penalty": 1.0,
                "top_k": 10,
                "top_p": 0.95,
            }
        )
        configs = {
            "greedy": greedy_config,
            "sample": sample_config,
        }
        return configs
