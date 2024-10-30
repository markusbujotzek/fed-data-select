import torch
import random
from torch.utils.data import Subset, DataLoader, random_split, ConcatDataset


class KSLoss:
    def __init__(
        self,
        benchmark_split_percentage: float = 0.05,
        benchmark_train_ratio: float = 0.8,
        batch_size: int = 32,
        benchmark_model=None,
        loss_fn=None,
    ):
        self.benchmark_split_percentage = benchmark_split_percentage
        self.benchmark_train_ratio = benchmark_train_ratio
        self.batch_size = batch_size
        self.benchmark_model = benchmark_model
        self.loss_fn = loss_fn

    def get_benchmark_data(self, train_dataloaders):
        """
        Extracts a percentage of samples from each federated client's DataLoader to form a benchmark dataset.

        Parameters
        ----------
        train_dataloaders : list of DataLoader
            List of DataLoaders, one per federated client.

        Returns
        -------
        train_dataloaders : list of DataLoader
            Updated train DataLoaders for each client with benchmark samples removed.
        benchmark_dataset : ConcatDataset
            Dataset containing benchmark samples from each client.
        """
        benchmark_samples = []

        for idx, data_loader in enumerate(train_dataloaders):
            total_samples = len(data_loader.dataset)
            num_benchmark_samples = int(total_samples * self.benchmark_split_percentage)

            # Select random indices for benchmark
            all_indices = list(range(total_samples))
            benchmark_indices = random.sample(all_indices, num_benchmark_samples)
            remaining_indices = list(set(all_indices) - set(benchmark_indices))

            # Collect benchmark samples and update client DataLoader
            benchmark_samples.extend(
                [data_loader.dataset[i] for i in benchmark_indices]
            )
            remaining_subset = Subset(data_loader.dataset, remaining_indices)
            train_dataloaders[idx] = DataLoader(
                remaining_subset, batch_size=self.batch_size, shuffle=True
            )

        # Combine samples into a single benchmark dataset
        benchmark_dataset = ConcatDataset(benchmark_samples)
        return train_dataloaders, benchmark_dataset

    def split_benchmark_data(self, benchmark_dataset):
        """
        Splits the benchmark dataset into training and testing datasets.

        Parameters
        ----------
        benchmark_dataset : ConcatDataset
            Combined dataset containing samples from each client.

        Returns
        -------
        benchmark_train_loader : DataLoader
            DataLoader for benchmark training data.
        benchmark_test_loader : DataLoader
            DataLoader for benchmark testing data.
        """
        num_train = int(len(benchmark_dataset) * self.benchmark_train_ratio)
        num_test = len(benchmark_dataset) - num_train
        benchmark_train_dataset, benchmark_test_dataset = random_split(
            benchmark_dataset, [num_train, num_test]
        )

        benchmark_train_loader = DataLoader(
            benchmark_train_dataset, batch_size=self.batch_size, shuffle=True
        )
        benchmark_test_loader = DataLoader(
            benchmark_test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return benchmark_train_loader, benchmark_test_loader

    def train_benchmark_model(self, benchmark_train_loader):
        """
        Trains the benchmark model on the benchmark_train_loader data.

        Parameters
        ----------
        benchmark_train_loader : DataLoader
            DataLoader for training the benchmark model.

        Returns
        -------
        trained_model : Model
            Trained benchmark model.
        """
        trained_model = (
            self.benchmark_model
        )  # Assume benchmark_model is trained in place
        # ... Implement training logic here ...
        return trained_model

    def eval_benchmark_model(self, benchmark_model, data_loader):
        """
        Evaluates the benchmark model on the provided DataLoader and calculates loss per sample.

        Parameters
        ----------
        benchmark_model : Model
            Trained benchmark model to evaluate.
        data_loader : DataLoader
            DataLoader for which to compute sample losses.

        Returns
        -------
        sample_losses : list
            List of loss values per sample.
        """
        sample_losses = []
        benchmark_model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = benchmark_model(inputs)
                loss = torch.nn.functional.mse_loss(outputs, labels, reduction="none")
                sample_losses.extend(loss.tolist())
        return sample_losses

    def select_train_samples(self, train_sample_benchmark_loss):
        """
        Selects a subset of samples from train data based on loss values.

        Parameters
        ----------
        train_sample_benchmark_loss : list
            List of loss values for each training sample.

        Returns
        -------
        kept_indices : list
            Indices of samples to keep based on loss values.
        """
        # Sort by loss values (ascending) and keep a specified portion
        sorted_train_losses = sorted(
            enumerate(train_sample_benchmark_loss), key=lambda x: x[1]
        )
        # TODO: Implement real KSLoss dataselection!
        # Example: Keep the samples with lowest losses
        kept_indices = [
            idx for idx, _ in sorted_train_losses[: len(sorted_train_losses) // 2]
        ]
        return kept_indices

    def ksloss_data_selection(self, train_dataloaders):
        """
        Interface of KSLoss to federated training.
        This function gets the unmodified trainloaders of the federated training, one trainloader per FL client.
        Splits from the trainloaders a small percentage and saves this small split of in one benchmark dataset via def get_benchmark_data().
        This benchmark dataset is again split by a ceratin ratio into a benchmark_train and benchmark_test dataloader via def split_benchmark_data().
        The benchmark_train dataloader is used to train the benchmark model w_benchmark via def train_benchmark_model().
        The data of benchmark_test dataloader is fed through the benchmark model w_benchmark and computed loss values are saved per data sample in benchmark_test_sample_benchmark_loss via def eval_benchmark_model().
        The data of all train dataloader are fed through the benchmark model w_benchmark and computed loss values are saved per data sample in train_sample_benchmark_losscia compute trainset_eval_benchmark_model().
        Sort train_sample_benchmark_loss by the computed loss values.
        The loss lists benchmark_test_sample_benchmark_loss and train_sample_benchmark_loss are given to a function to finally select the kept train samples via def select_train_smaples().
        Only the kept train samples should stay in the individual train_dataloaders.
        """
        # Step 1: Extract benchmark data
        train_dataloaders, benchmark_dataset = self.get_benchmark_data(
            train_dataloaders
        )

        # Step 2: Split benchmark dataset into train and test sets
        benchmark_train_loader, benchmark_test_loader = self.split_benchmark_data(
            benchmark_dataset
        )

        # Step 3: Train benchmark model
        trained_benchmark_model = self.train_benchmark_model(benchmark_train_loader)

        # Step 4: Evaluate on benchmark test set
        benchmark_test_sample_benchmark_loss = self.eval_benchmark_model(
            trained_benchmark_model, benchmark_test_loader
        )

        # Step 5: Evaluate on each client train dataset
        train_sample_benchmark_loss = []
        for data_loader in train_dataloaders:
            sample_losses = self.eval_benchmark_model(
                trained_benchmark_model, data_loader
            )
            train_sample_benchmark_loss.extend(sample_losses)

        # Step 6: Select samples to keep based on sorted loss values
        kept_indices = self.select_train_samples(train_sample_benchmark_loss)

        # Step 7: Filter each train dataloader to retain only kept samples
        for i, data_loader in enumerate(train_dataloaders):
            all_indices = list(range(len(data_loader.dataset)))
            retained_indices = [idx for idx in all_indices if idx in kept_indices]
            retained_subset = Subset(data_loader.dataset, retained_indices)
            train_dataloaders[i] = DataLoader(
                retained_subset, batch_size=self.batch_size, shuffle=True
            )

        return train_dataloaders
