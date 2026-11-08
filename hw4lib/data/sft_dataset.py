"""Supervised prompt/response examples for HW4P2."""
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from ..utils.webnlg import format_webnlg_prompt


class SFTDataset(Dataset):
    """Build response-only next-token targets from grouped WebNLG records.

    Args:
        records: Inputs with id, triples, and a list of references.
        tokenizer: P1 tokenizer with encode() and eos_id.
        max_length: Maximum number of input tokens, excluding the shifted EOS.
        expand_references: Train on all references; use the first for dev CE.
    """
    def __init__(self, records, tokenizer, max_length, *, expand_references=True):
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise ValueError("max_length must be positive")
        if not records:
            raise ValueError("The dataset must contain examples")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        for record in records:
            refs = record.get("references")
            if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or not r.strip() for r in refs):
                raise ValueError("Every labeled input needs nonempty references")
            prompt = tokenizer.encode(format_webnlg_prompt(record["triples"]))
            for response in refs if expand_references else refs[:1]:
                answer = tokenizer.encode(response)
                if not prompt or not answer:
                    raise ValueError("Prompt and response must contain tokens")
                if len(prompt) + len(answer) > max_length:
                    raise ValueError(f"Input {record['id']} exceeds context {max_length}; no text was truncated")
                self.examples.append(self._make_example(prompt, answer))

    def _make_example(self, prompt, answer):
        """Return shifted input_ids and labels; supervise response and EOS only."""
        # TODO: Join prompt, response, and EOS. Shift once for next-token prediction.
        # The first response target sits at index len(prompt) - 1.
        tokens = list(prompt) + list(answer) + [self.tokenizer.eos_id]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        labels[: len(prompt) - 1] = -100
        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        # TODO: Return the number of prompt/response pairs.
        return len(self.examples)

    def __getitem__(self, index):
        # TODO: Return one example without allowing the caller to alter stored tensors.
        example = self.examples[index]
        return {key: value.clone() for key, value in example.items()}

    def collate_fn(self, examples):
        """Right-pad inputs with EOS, and labels with -100 (not EOS)."""
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        # TODO: Pad each field to the longest example in this batch.
        batch = {}
        for key in examples[0]:
            padding_value = -100 if key == "labels" else self.tokenizer.eos_id
            batch[key] = pad_sequence(
                [example[key] for example in examples],
                batch_first=True,
                padding_value=padding_value,
            )
        return batch