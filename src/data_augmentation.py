
import numpy as np
import random
from collections import Counter

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class BalancedAugmentedDataset(Dataset):
    """
    Balance KL classes using small-angle rotation augmentation.
    Original images are retained.
    Additional examples are generated dynamically.
    """

    def __init__(self, samples, image_size=224,
                 max_rotation=5, seed=42,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):

        self.original_samples = [
            (str(path), int(label))
            for path, label in samples
        ]

        self.max_rotation = max_rotation

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        counts = Counter(
            label for _, label in self.original_samples
        )

        if not counts:
            raise ValueError("Training dataset is empty.")

        target_count = max(counts.values())
        rng = random.Random(seed)

        # Store original images without augmentation
        self.samples = [
            (path, label, False)
            for path, label in self.original_samples
        ]

        # Add augmented examples for minority classes
        for label in sorted(counts):

            class_samples = [
                path for path, y in self.original_samples
                if y == label
            ]

            needed = target_count - len(class_samples)

            # Cycle through minority images to distribute
            # augmentation approximately evenly
            for i in range(needed):
                path = class_samples[i % len(class_samples)]
                self.samples.append((path, label, True))

        rng.shuffle(self.samples)

        print("Original distribution:", dict(sorted(counts.items())))
        print("Balanced distribution:",
              dict(sorted(Counter(
                  label for _, label, _ in self.samples
              ).items())))
        print("Total training examples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label, augment = self.samples[idx]
        image = default_loader(path)

        if augment:
            # Remember original dimensions
            original_width, original_height = image.size

            # Reflect padding before rotation
            padding = max(
                8,
                round(min(original_width, original_height) * 0.10)
            )

            image = TF.pad(
                image,
                padding=padding,
                padding_mode="reflect"
            )

            # Random rotation between -5 and +5 degrees
            angle = random.uniform(
                -self.max_rotation,
                self.max_rotation
            )

            image = TF.rotate(
                image,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR
            )

            # Crop back to original dimensions
            image = TF.center_crop(
                image,
                output_size=(original_height, original_width)
            )

        # Resize, convert to tensor, and normalize
        image = self.transform(image)

        return image, label
