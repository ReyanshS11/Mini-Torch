from typing import Iterable

from mini_torch.core.tensor import Tensor
class Dataset:
    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx) -> Iterable:
        return Tensor([]), Tensor([])