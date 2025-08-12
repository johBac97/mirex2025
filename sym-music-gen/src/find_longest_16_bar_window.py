from pathlib import Path

import numpy as np

import tqdm

import symusic
import miditok

def longest_16bar_window(midi_files, tokenizer) -> int:
    if isinstance(midi_files, (str, Path)):
        midi_files = list(Path(midi_files).glob("**/*.mid"))
    max_len = 0
    for f in tqdm.tqdm(midi_files):
        try:
            score = symusic.Score(f)
            ids = np.array(tokenizer.encode(score)[0].ids)
            bar = tokenizer.vocab["Bar_None"]
            bar_starts = np.where(ids == bar)[0]
            # slide a 16-bar window
            for i in range(0, len(bar_starts) - 16):
                L = bar_starts[i + 16] - bar_starts[i]
                if L > max_len:
                    max_len = L
        except Exception:
            continue
    return int(max_len)


if __name__ == '__main__':
    config = miditok.TokenizerConfig(
        pitch_range=(0, 127),
        use_velocities=False,
        encode_ids_splits="no",
        use_pitchdrum_tokens=False,
    )
    tokenizer = miditok.REMI(config)

    midi_files = Path('../data/filtered_aria/train')
    print(longest_16bar_window(midi_files, tokenizer))
