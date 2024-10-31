import torch
import random
from torch.utils.data import Subset, DataLoader, random_split, ConcatDataset

from flamby.strategies.utils import _Model, DataLoaderWithMemory
from utils.utils import DatasetWrapper, DatasetWithIDWrapper


class KSLoss:
    def __init__(
        self,
        benchmark_split_percentage: float = 0.05,
        benchmark_train_ratio: float = 0.8,
        batch_size: int = 32,
        benchmark_model=None,
        loss_fn=None,
        args=None,
    ):
        self.benchmark_split_percentage = benchmark_split_percentage
        self.benchmark_train_ratio = benchmark_train_ratio
        self.batch_size = batch_size
        self.benchmark_model = benchmark_model
        self.loss_fn = loss_fn
        self.args = args

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
            print(
                f"Dataset {idx} contains {total_samples} samples BEFORE extraction of benchmark split!"
            )
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
            print(
                f"Dataset {idx} contains {len(train_dataloaders[idx].dataset)} samples AFTER extraction of benchmark split!"
            )

        # Combine samples into a single benchmark dataset
        # benchmark_dataset = ConcatDataset(benchmark_samples)
        # benchmark_dataset = ConcatDataset([Subset(train_dataloaders[idx].dataset, benchmark_indices) for idx, benchmark_indices in enumerate(benchmark_samples)])
        benchmark_dataset = ConcatDataset(
            [DatasetWrapper(sample) for sample in benchmark_samples]
        )
        print(f"Benchmark dataset conatins {len(benchmark_dataset)} samples!")
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
        benchmark_train_indices, benchmark_test_indices = random_split(
            range(len(benchmark_dataset)), [num_train, num_test]
        )

        # Convert to actual samples instead of indices
        benchmark_train_samples = [
            benchmark_dataset[i] for i in benchmark_train_indices.indices
        ]
        benchmark_test_samples = [
            benchmark_dataset[i] for i in benchmark_test_indices.indices
        ]

        # Create Subsets based on the actual samples
        benchmark_train_dataset = ConcatDataset(
            [DatasetWrapper(sample) for sample in benchmark_train_samples]
        )
        benchmark_test_dataset = ConcatDataset(
            [DatasetWrapper(sample) for sample in benchmark_test_samples]
        )

        benchmark_train_loader = DataLoader(
            benchmark_train_dataset, batch_size=self.batch_size, shuffle=True
        )
        benchmark_test_loader = DataLoader(
            benchmark_test_dataset, batch_size=1, shuffle=False
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
        benchmark_train_loader_with_memory = DataLoaderWithMemory(
            benchmark_train_loader
        )

        # Create a new instance of _Model for the benchmark_model
        model_instance = _Model(
            model=self.benchmark_model,
            train_dl=benchmark_train_loader_with_memory,
            optimizer_class=self.args[
                "optimizer_class"
            ],  # or another optimizer you prefer
            lr=self.args["learning_rate"],  # Set your desired learning rate
            loss=self.loss_fn,  # Replace with your actual loss function
            nrounds=100,  # This should be the number of local updates, e.g. self.args["nrounds"]
            log=False,  # Set to True if you want to log training
        )

        # Train the benchmark model locally using the provided DataLoader
        model_instance._local_train(benchmark_train_loader_with_memory, 100)

        # Return the trained benchmark model
        return self.benchmark_model

    def convert_dataloader_to_dataloader_w_id(
        self, list_of_dataloaders: DataLoader, batch_sizes: list = None
    ):

        # handle if batch_sizes is None
        if not batch_sizes:
            batch_sizes = [None] * len(list_of_dataloaders)

        dataloaders_w_id = []
        for idx, dataloader in enumerate(list_of_dataloaders):
            ds = dataloader.dataset
            ds_w_id = DatasetWithIDWrapper(ds)
            dataloader_w_id = DataLoader(
                ds_w_id,
                batch_size=1 if batch_sizes[idx] is None else batch_sizes[idx],
                shuffle=dataloader.shuffle if "shuffle" in dataloader else False,
            )
            dataloaders_w_id.append(dataloader_w_id)
        return dataloaders_w_id

    def compute_sample_losses(self, model, data_loader):
        """
        Calculate the loss for each data sample by passing the data through the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model used for evaluation.
        data_loader : torch.utils.data.DataLoader
            The DataLoader providing the input data and labels.

        Returns
        -------
        list of dict
            A list containing dictionaries that map sample IDs to their corresponding loss values.
        """
        sample_losses = []
        model.eval()
        with torch.no_grad():
            for inputs, labels, id in data_loader:
                outputs = model(inputs)
                loss = self.args["loss"].forward(outputs, labels)
                sample_losses.append({id[0]: loss.item()})

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

        # Step 4: add hash id to dataloaders to clearly identify every sample
        benchmark_test_loader_w_id = self.convert_dataloader_to_dataloader_w_id(
            [benchmark_test_loader]
        )[0]
        train_dataloaders_w_id = self.convert_dataloader_to_dataloader_w_id(
            train_dataloaders
        )

        # Step 5: Evaluate on benchmark test set
        benchmark_test_sample_benchmark_loss = self.compute_sample_losses(
            trained_benchmark_model, benchmark_test_loader_w_id
        )

        # Step 6: Evaluate on each client train datasets
        train_sample_benchmark_loss_per_client = []
        for train_dataloader_w_id in train_dataloaders_w_id:
            train_sample_benchmark_loss_per_client.append(
                self.compute_sample_losses(
                    trained_benchmark_model, train_dataloader_w_id
                )
            )

        # Step 7: Summarize all client's training samples and sort w.r.t. their loss values in ascending order
        # add all client's data into one id_loss_collection
        train_sample_benchmark_loss_all_clients = sum(
            train_sample_benchmark_loss_per_client, []
        )
        # also make sure the hashes are indeed unique
        keys = [
            key for d in train_sample_benchmark_loss_all_clients for key in d.keys()
        ]
        assert len(keys) == len(set(keys)), "Duplicate keys found."
        # sort in ascending order
        sorted_train_sample_benchmark_loss_all_clients = sorted(
            train_sample_benchmark_loss_all_clients, key=lambda d: list(d.values())[0]
        )

        # Step 6: Select samples to keep based on sorted loss values
        kept_indices = self.select_train_samples(train_sample_benchmark_loss)

        # # Step 7: Filter each train dataloader to retain only kept samples
        # for i, data_loader in enumerate(train_dataloaders):
        #     all_indices = list(range(len(data_loader.dataset)))
        #     retained_indices = [idx for idx in all_indices if idx in kept_indices]
        #     retained_subset = Subset(data_loader.dataset, retained_indices)
        #     train_dataloaders[i] = DataLoader(
        #         retained_subset, batch_size=self.batch_size, shuffle=True
        #     )

        return train_dataloaders
