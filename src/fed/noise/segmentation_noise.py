import random
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from monai.transforms.transform import Transform


def closed_set_label_flip(labels: list = None, noisy_indices: list = None):
    """
    Applies closed-set label flipping to a given list of label tensors, introducing controlled noise
    by replacing foreground class labels with different ones from the same set.

    Args:
        labels (list(torch.Tensor)): A list of original label tensors where each tensor represents
            a multi-class segmentation mask in one-hot encoded format.
        noisy_indices (list): A list of indices indicating which label tensors in `labels` should
            be modified to introduce noise.

    Returns:
        list(torch.Tensor): The updated list of label tensors, where specified indices have been
            modified with noisy labels.

    Notes:
        - Background is assumed to be represented by the label 0 and is not modified.
        - Foreground labels are randomly replaced with other foreground labels while ensuring they
          are different from the original label.
        - This function only modifies tensors that have more than one foreground class.
    """
    # get foreground class indices
    # Works for sure but computationally expensive
    # fg_labels = []
    # for idx in range(0, len(labels)-1):
    #     non_onehot_mask = torch.argmax(labels[idx], dim=0)
    #     fg_labels.extend([int(x) for x in torch.unique(non_onehot_mask).tolist() if x != 0])
    # foreground_class_labels = list(set(fg_labels))
    # Computational less expensive
    # Collapse the tensor over all samples and spatial dimensions at once
    collapsed_labels = torch.argmax(
        labels, dim=1
    )  # Shape [N, D, H, W] -> Non-one-hot format per sample
    unique_labels = torch.unique(
        collapsed_labels
    )  # Get unique class indices across the entire dataset
    # Filter out the background class (assuming background is labeled as 0)
    foreground_class_labels = [
        int(label) for label in unique_labels.tolist() if label != 0
    ]

    noisy_samples_generated = 0
    for idx in noisy_indices:
        # check if sample has more than 2 classes (bg + N * fg)
        if labels[idx].size()[0] > 2:
            # get the non-one-hot-encoded mask and its forground class labels
            org_mask = torch.argmax(labels[idx], dim=0)
            org_mask_foreground_class_labels = [
                int(x) for x in torch.unique(org_mask).tolist()
            ]
            org_mask_foreground_class_labels.remove(0)

            # randomly select a new closed-set class label for each foreground class label
            noisy_mask = org_mask
            for org_mask_foreground_class_label in org_mask_foreground_class_labels:
                noisy_mask_class_label = random.choice(
                    [
                        label
                        for label in foreground_class_labels
                        if label != org_mask_foreground_class_label
                    ]
                )
                noisy_mask = torch.where(
                    org_mask == org_mask_foreground_class_label,
                    noisy_mask_class_label,
                    org_mask,
                )

            # convert noisy mask to one-hot encoding
            noisy_mask_one_hot = F.one_hot(
                noisy_mask, num_classes=(len(org_mask_foreground_class_labels) + 1)
            )
            noisy_mask_one_hot = noisy_mask_one_hot.permute(3, 0, 1, 2)
            labels[idx] = noisy_mask_one_hot

            noisy_samples_generated += 1
        else:
            print(
                "WARNING: Noisy mask not generated for sample with single foreground class label."
            )

    print(
        f"{noisy_samples_generated} noisy samples out of {len(noisy_indices)} requested noisy samples generated."
    )

    return labels


def add_noise(
    noise_type: str = None, dataloader: DataLoader = None, noise_percentage: float = 0.0
):
    """
    Applies label noise to a given percentage of samples in a dataloader.

    Args:
        noise_type (str): The type of noise to apply. Default is None. Options are "closed_set_label_flip", "open_set_label_flip".
        dataloader (DataLoader): The org dataloader.
        noise_percentage (float): The percentage of labels to be modified (0.0 to 1.0).

    Returns:
        DataLoader: A new dataloader with label noise applied.
    """
    # Extract all data and labels from the org dataloader
    data = []
    labels = []
    for batch in dataloader:
        inputs, targets, _ = batch
        data.append(inputs)
        labels.append(targets)

    # Concatenate all batches into full tensors
    data = torch.cat(data)
    labels = torch.cat(labels)

    # Calculate the number of samples to apply noise to
    num_samples = len(labels)
    num_noisy_samples = math.ceil(noise_percentage * num_samples)
    print(f"Applying noise to {num_noisy_samples} out of {num_samples} samples.")

    # Select random indices for applying noise
    noisy_indices = random.sample(range(num_samples), num_noisy_samples)

    if noise_type == "closed_set_label_flip":
        labels = closed_set_label_flip(labels, noisy_indices)
    # elif noise_type == "open_set_label_flip":
    #     labels = open_set_label_flip(labels, noisy_indices)
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")

    # TODO: proper handling of new noise samples with existing dataloader and its dataset
    # Create a new TensorDataset with noisy labels
    noisy_dataset = TensorDataset(data, labels)

    # Return a new dataloader with the modified dataset
    return DataLoader(
        noisy_dataset, batch_size=dataloader.batch_size, shuffle=dataloader.shuffle
    )


class ClosedSetLabelSwap(Transform):
    def __init__(
        self,
        label_swap_type: str = None,
        dataset_class_labels: list = None,
    ):
        super().__init__()
        self.label_swap_type = label_swap_type
        self.dataset_class_labels = dataset_class_labels
        if self.label_swap_type == "closed_set_label_flip":
            self.possible_noisy_class_labels = self.dataset_class_labels
        elif self.label_swap_type == "open_set_label_flip":
            self.possible_noisy_class_labels = self.dataset_class_labels + list(
                range(
                    max(self.dataset_class_labels) + 1,
                    max(self.dataset_class_labels) * 2 + 1,
                )
            )
            self.possible_noisy_class_labels.remove(0)

    def __call__(self, label):
        """
        Apply closed-set label flipping to a given label tensor.
        """
        # swap labels if closed_set but more than 2 classes or open_set
        if (label.size()[0] > 2 and self.label_swap_type == "closed_set") or (
            self.label_swap_type == "open_set"
        ):
            # get the non-one-hot-encoded mask and its forground class labels
            org_mask = torch.argmax(label, dim=0)
            org_mask_foreground_class_labels = [
                int(x) for x in torch.unique(org_mask).tolist() if x != 0
            ]

            # randomly select a new class label for each foreground class label
            noisy_mask = org_mask
            for org_mask_foreground_class_label in org_mask_foreground_class_labels:
                noisy_mask_class_label = random.choice(
                    [
                        label
                        for label in self.possible_noisy_class_labels
                        if label != org_mask_foreground_class_label
                    ]
                )
                noisy_mask = torch.where(
                    org_mask == org_mask_foreground_class_label,
                    noisy_mask_class_label,
                    org_mask,
                )

            # convert noisy mask to one-hot encoding
            noisy_mask_one_hot = F.one_hot(
                noisy_mask, num_classes=(len(org_mask_foreground_class_labels) + 1)
            )
            noisy_mask_one_hot = noisy_mask_one_hot.permute(3, 0, 1, 2)
            return noisy_mask_one_hot
        else:
            print(
                "WARNING: Noisy mask not generated for sample with single foreground class label."
            )
            return label
