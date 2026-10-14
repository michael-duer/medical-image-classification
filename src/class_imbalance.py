
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms


def get_transforms(image_size=224, train=True,
                   mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):

    steps = [transforms.Resize((image_size, image_size))]

    if train:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transforms.Compose(steps)


def create_weighted_sampler(samples, num_classes):
    """Calculate inverse-frequency sampling weights."""

    labels = np.array([label for path, label in samples])

    counts = np.bincount(labels, minlength=num_classes)

    if np.any(counts == 0):
        raise ValueError("One or more training classes are empty.")

    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True
    )

    print("Training class counts:", counts)
    print("Class sampling weights:", class_weights)

    return sampler
