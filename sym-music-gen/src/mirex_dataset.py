from pathlib import Path
import numpy as np
import torch


class MIREXCustomDataset(torch.utils.data.Dataset):
    def __init__(self, midi_files: Path | list[Path], tokenizer):
        super().__init__()
        self._midi_files = (
            midi_files
            if isinstance(midi_files, list)
            else list(midi_files.glob("**/*.mid"))
        )

        self._tokenizer = tokenizer

        self._num_prompt_measures = 4
        self._num_completion_measures = 12

    def __len__(self):
        return len(self._midi_files)

    def __getitem__(self, idx: int):
        file_path = self._midi_files[idx]

        encoding = self._tokenizer.encode(file_path)[0]

        token_ids = np.array(encoding.ids)

        # Select random bar to start
        bar_starts = np.where(token_ids == self._tokenizer.vocab["Bar_None"])[0]

        selected_bar_start = np.random.randint(0, len(bar_starts) - 16)
        sample_start = bar_starts[selected_bar_start]
        sample_end = bar_starts[selected_bar_start + 16]

        sample = token_ids[sample_start:sample_end]

        input_ids = torch.from_numpy(sample)
        attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()

        # Get the start of the completion measures
        index_start_completion = torch.where(
            labels == self._tokenizer.vocab["Bar_None"]
        )[0][self._num_prompt_measures].item()

        labels[0:index_start_completion] = -100

        print(f"MAx:\t{input_ids.max()}")
        return {
            "input_ids": input_ids,
            # "attention_mask": attention_mask,
            "labels": labels,
        }
