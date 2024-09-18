import numpy as np


class CastToPrecision:
    def __init__(self, precision: int = 16):
        self.precision = precision

    def __call__(self, tensor: np.ndarray):
        if self.precision == 16:
            return tensor.astype(np.float16)

        elif self.precision == 32:
            return tensor.astype(np.float32)

        elif self.precision == 64:
            return tensor.astype(np.float64)
