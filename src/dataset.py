import random
import numpy as np
from torchvision import datasets, transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def train_tf(size):
    # NOTE: no horizontal flip - a mirrored muzzle pattern is a different pattern.
    return T.Compose([
        T.Resize((size + 32, size + 32)),
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.RandomRotation(12),
        T.ColorJitter(0.4, 0.4, 0.3, 0.05),
        T.RandomGrayscale(0.1),
        T.ToTensor(), T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def eval_tf(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


def make_pairs(labels, n_pairs=5000, seed=0):
    """Random positive / negative index pairs for verification metrics."""
    rng = random.Random(seed)
    labels = np.asarray(labels)
    by_cls = {}
    for i, l in enumerate(labels):
        by_cls.setdefault(int(l), []).append(i)
    multi = [c for c, v in by_cls.items() if len(v) >= 2]
    classes = list(by_cls)
    pos, neg = [], []
    for _ in range(n_pairs):
        c = rng.choice(multi)
        pos.append(tuple(rng.sample(by_cls[c], 2)))
        c1, c2 = rng.sample(classes, 2)
        neg.append((rng.choice(by_cls[c1]), rng.choice(by_cls[c2])))
    return pos, neg
