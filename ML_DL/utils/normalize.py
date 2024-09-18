import numpy as np


class NormalizeToOne:

    def __call__(self, tensor: np.ndarray):
        return (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor) + 1e-8)


if __name__ == "__main__":
    print(1)
