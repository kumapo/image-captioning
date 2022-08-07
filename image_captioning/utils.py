import os

def seed_everything(seed: int = 42):
    import random
    import torch
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


import argparse
import json
import pathlib

class PathSerializableJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, pathlib.Path):
            return str(o)
        elif isinstance(o, set):
            return list(o)
        return super().default(o)


def save_args(args: argparse.Namespace, dst: pathlib.Path):
    if dst is None:
        return
    with open(dst, 'w') as f:
        json.dump(vars(args), f, cls=PathSerializableJSONEncoder)


def load_args(src: pathlib.Path) -> argparse.Namespace:
    if src is not None and src.exists():
        with open(src, 'r') as f:
            ns = json.load(f)
        return argparse.Namespace(**ns)
