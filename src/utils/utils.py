import torch
import hashlib
from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample


class DatasetWithIDWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        Wraps an existing dataset to add unique identifiers for each element.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The original dataset to wrap.
        """
        self.dataset = dataset
        self.ids = self._generate_ids()

    def _generate_ids(self):
        """
        Generates a unique ID for each element in the dataset.

        Returns
        -------
        list
            List of unique identifiers for each data element.
        """
        ids = []
        for i in range(len(self.dataset)):
            # Use a hash of the index combined with the data's string representation
            item = self.dataset[i]
            item_data = str(item[0])  # Assuming first element in item is data
            item_hash = hashlib.md5(f"{i}_{item_data}".encode()).hexdigest()
            ids.append(item_hash)
        return ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns the data element, label, and unique identifier.

        Parameters
        ----------
        idx : int
            Index of the data element to retrieve.

        Returns
        -------
        tuple
            (data element, label, unique identifier)
        """
        item = self.dataset[idx]
        unique_id = self.ids[idx]
        return (*item, unique_id)

    def get_item_by_id(self, unique_id):
        """
        Retrieve the data element and label using the unique identifier.

        Parameters
        ----------
        unique_id : str
            The unique identifier for the data element.

        Returns
        -------
        tuple
            (data element, label)
            If the unique ID is found; otherwise, raises ValueError.
        """
        if unique_id in self.ids:
            idx = self.ids.index(unique_id)
            return self.dataset[idx]  # Return the data and label tuple
        else:
            raise ValueError(f"Unique ID {unique_id} not found in the dataset.")
