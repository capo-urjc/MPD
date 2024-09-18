from functools import reduce
import pandas as pd


def load_correspondences(path_correspondences: str) -> dict:
    dict_correspondences: dict = pd.read_csv(path_correspondences).to_dict(orient='list')
    return dict_correspondences


def unique(list1: list):
    """

    :param list1: list with possible duplicates

    :return:the function returns the list passed as an argument without duplica>
    """
    ans = reduce(lambda re, x: re + [x] if x not in re else re, list1, [])
    return ans
