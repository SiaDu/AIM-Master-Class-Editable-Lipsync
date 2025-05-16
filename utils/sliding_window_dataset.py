from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class SlidingWindowLipsyncDataset(Dataset):
    """Generate fixed‑length sliding‑window pairs (audio, expression) for lip‑sync training.

    Parameters
    ----------
    audio_list : list of np.ndarray | torch.Tensor
        Each element is a sequence of shape (T, 464).
    expr_list : list of np.ndarray | torch.Tensor
        Each element is a sequence of shape (T, 284).
    window_size : int, default 60
        Number of frames per sample window.
    stride : int, default 10
        Step size to slide the window.
    """

    def __init__(
        self,
        audio_list: Sequence,
        expr_list: Sequence,
        window_size: int = 60,
        stride: int = 10,
    ) -> None:
        if len(audio_list) != len(expr_list):
            raise ValueError("audio_list and expr_list must be the same length")

        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        win, step = window_size, stride

        for audio_seq, expr_seq in zip(audio_list, expr_list):
            T = min(len(audio_seq), len(expr_seq))
            for start in range(0, T - win + 1, step):
                end = start + win
                a_chunk = torch.as_tensor(audio_seq[start:end], dtype=torch.float32)
                e_chunk = torch.as_tensor(expr_seq[start:end], dtype=torch.float32)
                self.samples.append((a_chunk, e_chunk))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]
