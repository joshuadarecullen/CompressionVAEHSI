import numpy as np
from typing import Tuple, List


class Split:
    def __init__(self, dataset_size: int, trainvaltest: Tuple) -> None:
        self.dataset_size = dataset_size
        self.trainvaltest = trainvaltest

    def create_splits(self) -> List:
        return 0
