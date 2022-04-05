import os
import numpy as np


def number_of_files(path: str):
    return len(os.listdir(path))


def select_indices(margin, select, total):
    return list(np.linspace(margin, total - margin, select, dtype=int))


def offset_indices(indices, offset):
    return [ind + offset for ind in indices]


def join_lists(a, b):
    return [*a, *b]
