"""
Multi-task DataLoader for ChemDiffuse3D.

Implements weighted random sampling across multiple task-specific
dataloaders with infinite cycling.
"""

import random


class MultiTaskDataLoader:
    """
    Samples batches from multiple DataLoaders using weighted random selection.

    Each DataLoader is cycled infinitely, and at each step, one task
    is randomly selected according to the provided sampling weights.

    Args:
        loaders (dict): Mapping from task_id to DataLoader
        sampling_weights (list): Probability weights for each task
    """

    def __init__(self, loaders, sampling_weights):
        self.loaders = [iter(self.infinite_cycle(loaders[key])) for key in loaders]
        self.weights = sampling_weights

        for key, loader in loaders.items():
            if hasattr(loader, "rng_types"):
                loader.rng_types = None

    def infinite_cycle(self, loader):
        """Cycle through a DataLoader infinitely."""
        while True:
            for batch in loader:
                yield batch

    def __iter__(self):
        return self

    def __next__(self):
        task_id = random.choices(range(len(self.loaders)), weights=self.weights, k=1)[0]
        batch = next(self.loaders[task_id])
        return task_id, batch
