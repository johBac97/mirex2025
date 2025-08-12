import torch
from torch.nn.utils.rnn import pad_sequence

def make_collate_fn(pad_id: int, fixed_len: int | None = None, return_mask: bool = False):
    """Pads input_ids to batch max (or fixed_len), pads labels with -100 (ignored by CE)."""
    def _collate(batch):
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        attn = [b["attention_mask"] for b in batch]

        if fixed_len is None:
            T = max(x.size(0) for x in input_ids)
        else:
            T = fixed_len

        def pad_to(x, value):
            if x.size(0) < T:
                pad = (0, T - x.size(0))
                return torch.nn.functional.pad(x, pad, value=value)
            else:
                return x[:T]  # truncate if needed

        input_ids = torch.stack([pad_to(x, pad_id) for x in input_ids], dim=0)
        # labels: pad with -100 so cross-entropy ignores them
        labels = torch.stack([pad_to(x, -100) for x in labels], dim=0)
        # attention mask: 1 for real tokens, 0 for pad
        attn = torch.stack([pad_to(x, 0) for x in attn], dim=0).to(torch.long)

        # Your `training_step` expects (idx, targets) when args.my_qa_mask != 1
        return (input_ids, labels) if not return_mask else (input_ids, labels, attn)
    return _collate
