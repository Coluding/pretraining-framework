from dataclasses import dataclass, field
from transformers import DataCollatorForLanguageModeling
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import *


@dataclass
class CollatorConfig:
    tokenizer: Tokenizer
    is_downstream: bool = False
    max_length: Optional[int] = None
    pad_token_id: int = 0
    text_column: Optional[str] = "text"
    label_column: Optional[str] = None
    input_format: Union[Dict[str, List], None] = None

    def __post_init__(self):
        if self.input_format is None:
            self.input_format = {"input_ids": [],
                                 "position_ids": [],
                                 "sequence_ids": []}


class BaseCollator:
    def __init__(self, cfg: CollatorConfig):
        self.cfg = cfg

    def __call__(self, batch: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        inputs: Dict[str, torch.Tensor] =  self._tensorize_batch(batch)

        if self.cfg.is_downstream:
            inputs["labels"] = torch.tensor([item[self.cfg.label_column] for item in batch], dtype=torch.long)

        return inputs

    def _tensorize_batch(self, batch: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        inputs: Dict[str, list] = self.cfg.input_format

        for key in batch[0].keys():
            if key not in inputs.keys() and self.cfg.text_column is None:
                raise KeyError(f"Key {key} not found in input format. Check your input format in CollatorConfig")

        for key in ["input_ids", "position_ids", "sequence_ids"]:
            inputs[key] = [torch.tensor([item[key]], dtype=torch.long).squeeze() for item in batch]

            if self.cfg.max_length is not None:
                inputs[key].append(torch.zeros(self.cfg.max_length))

            inputs[key] = pad_sequence(inputs[key], batch_first=True, padding_value=self.cfg.pad_token_id)

            if self.cfg.max_length is not None:
                inputs[key] = inputs[key][:-1, :]

        return inputs

