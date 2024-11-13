import torch
import random
from torch.utils.data import Subset, DataLoader, random_split, ConcatDataset
import numpy as np
from scipy import stats
import copy

from flamby.strategies.utils import _Model, DataLoaderWithMemory
from flamby.datasets.fed_camelyon16 import collate_fn
from flamby.datasets.fed_camelyon16 import FedCamelyon16 as FedDataset
from flamby.datasets.fed_kits19 import softmax_helper


class KSLoss:
    def __init__(
        self,
        batch_size: int = 32,
        benchmark_model=None,
        loss_fn=None,
        metric=None,
        args=None,
    ):
        self.benchmark_ds_percentage = args["benchmark_ds_percentage"]
        self.benchmark_train_ratio = args["benchmark_train_ratio"]
        self.num_epochs_benchmark_model_training = args[
            "num_epochs_benchmark_model_training"
        ]
        self.batch_size = batch_size
        self.benchmark_model = benchmark_model
        self.loss_fn = loss_fn
        self.metric = metric
        self.lr = args["learning_rate"]
        self.optimizer = args["optimizer_class"](
            self.benchmark_model.parameters(), lr=self.lr
        )
        self.dataset = args["dataset"]
        self.args = args

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.benchmark_model.to(self.device)

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
        benchmark_ds_per_client = []

        for idx, data_loader in enumerate(train_dataloaders):
            total_samples = len(data_loader.dataset)
            num_benchmark_samples = int(total_samples * self.benchmark_ds_percentage)

            all_labels_in_benchmark_ds = False
            # for classification, i.e. label is an integer
            # labels_cache = [label.item() for _, labels, *_ in data_loader for label in labels]
            # for segmentation, i.e. label is a tensor
            labels_cache = []
            for _, labels, *_ in data_loader:
                labels_cache.extend(labels.tolist())
            labels_cached = list(torch.unique(torch.tensor(labels_cache)))
            all_labels = [t.item() for t in labels_cached]

            while not all_labels_in_benchmark_ds:
                # Select random indices for benchmark
                all_indices = np.arange(total_samples)
                benchmark_indices = np.random.choice(
                    all_indices, num_benchmark_samples, replace=False
                )
                remaining_indices = np.setdiff1d(all_indices, benchmark_indices)

                # for classification, i.e. label is an integer
                # Extract labels for benchmark samples without looping
                # benchmark_labels = set(data_loader.dataset[i][1].item() for i in benchmark_indices)
                benchmark_labels = np.unique(
                    list(
                        label.tolist()
                        for i in benchmark_indices
                        for label in [data_loader.dataset[i][1]]
                    )
                ).tolist()

                # Check if all labels are represented
                # all_labels = set(labels_cache)
                if all_labels == benchmark_labels:
                    all_labels_in_benchmark_ds = True

            # Collect benchmark samples and update client DataLoader
            benchmark_ds_per_client.append(
                Subset(data_loader.dataset, benchmark_indices)
            )
            remaining_subset = Subset(data_loader.dataset, remaining_indices)

            # Create DataLoader depending on the presence of `collate_fn`
            kwargs = {
                "batch_size": self.batch_size,
                "shuffle": True,
                "collate_fn": getattr(data_loader, "collate_fn", None),
            }
            train_dataloaders[idx] = DataLoader(
                remaining_subset, **{k: v for k, v in kwargs.items() if v is not None}
            )
            print(
                f"Dataset {idx} contains {total_samples} samples BEFORE extraction of benchmark split!"
            )
            print(
                f"Dataset {idx} contains {len(train_dataloaders[idx].dataset)} samples AFTER extraction of benchmark split!"
            )

        # Combine samples into a single benchmark dataset
        benchmark_dataset = ConcatDataset(benchmark_ds_per_client)

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

        benchmark_train_dataset = Subset(benchmark_dataset, benchmark_train_indices)
        benchmark_test_dataset = Subset(benchmark_dataset, benchmark_test_indices)

        if self.dataset == "FedCamelyon16":
            benchmark_train_loader = DataLoader(
                benchmark_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            benchmark_test_loader = DataLoader(
                benchmark_test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
        else:
            benchmark_train_loader = DataLoader(
                benchmark_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )
            benchmark_test_loader = DataLoader(
                benchmark_test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )

        print(
            f"Benchmark dataset split into {num_train} train and {num_test} test samples!"
        )
        return benchmark_train_loader, benchmark_test_loader

    # def train_benchmark_model_old(self, benchmark_train_loader, benchmark_test_loader):
    #     """
    #     Trains the benchmark model on the benchmark_train_loader data.

    #     Parameters
    #     ----------
    #     benchmark_train_loader : DataLoader
    #         DataLoader for training the benchmark model.

    #     Returns
    #     -------
    #     trained_model : Model
    #         Trained benchmark model.
    #     """
    #     print("Training benchmark model...")
    #     benchmark_train_loader_with_memory = DataLoaderWithMemory(
    #         benchmark_train_loader
    #     )

    #     # Create a new instance of _Model for the benchmark_model
    #     model_instance = _Model(
    #         model=self.benchmark_model,
    #         train_dl=benchmark_train_loader_with_memory,
    #         optimizer_class=self.args["optimizer_class"],
    #         lr=self.args["learning_rate"],
    #         loss=self.loss_fn,
    #         nrounds=self.num_epochs_benchmark_model_training,
    #         log=True,
    #         log_period=1,
    #     )

    #     # Train the benchmark model locally using the provided DataLoader
    #     model_instance._local_train(
    #         benchmark_train_loader_with_memory,
    #         num_updates=self.num_epochs_benchmark_model_training,
    #         validation_loader=benchmark_test_loader,
    #         metric=self.metric,
    #     )

    #     # Return the trained benchmark model
    #     print("Benchmark model training complete!")
    #     return self.benchmark_model

    def train_benchmark_model(self, benchmark_train_loader, benchmark_test_loader):
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
        print("Training benchmark model...")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        best_model_wts = copy.deepcopy(self.benchmark_model.state_dict())
        best_acc = 0.0
        self.benchmark_model = self.benchmark_model.to(device)
        # To draw loss and accuracy plots
        training_loss_list = []
        training_dice_list = []
        print(f" Benchmark Train Data Size: {len(benchmark_train_loader.dataset)}")
        print(f" Benchmark Test Data Size: {len(benchmark_test_loader.dataset)}")

        for epoch in range(self.num_epochs_benchmark_model_training):
            print(f"Epoch {epoch}/{self.num_epochs_benchmark_model_training - 1}")
            print("-" * 10)

            dice_list = []
            running_loss = 0.0
            dice_score = 0.0

            # set mode to training mode
            self.benchmark_model.train()
            # Iterate over data.
            for sample in benchmark_train_loader:
                inputs = sample[0].to(device)
                labels = sample[1].to(device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.benchmark_model(inputs)
                    loss = self.loss_fn(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    dice_score += self.metric(outputs, labels)

            # set mode to evaluation mode
            self.benchmark_model.eval()
            # Iterate over data.
            for sample in benchmark_test_loader:
                inputs = sample[0].to(device)
                labels = sample[1].to(device)

                with torch.set_grad_enabled(False):
                    preds_softmax = (
                        softmax_helper(outputs)
                        if self.dataset == "FedKits19"
                        else outputs
                    )
                    preds = preds_softmax.argmax(1)
                    dice_score = self.metric(preds.cpu(), labels.cpu())
                    dice_list.append(dice_score)

            epoch_loss = running_loss / len(benchmark_train_loader.dataset)
            epoch_acc = np.mean(dice_list)  # average dice

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.benchmark_model.state_dict())
                early_stopping_counter = 0
                print("Juhuuu! Best model updated!")
            if epoch_acc < best_acc:
                early_stopping_counter += 1
                print(
                    f"Ohooo! Early stopping counter increased to {early_stopping_counter}!"
                )
                if early_stopping_counter > 5:
                    print("Training early stopped!")
                    break

            print(
                "Training Loss: {:.4f} Validation Acc: {:.4f} ".format(
                    epoch_loss, epoch_acc
                )
            )
            training_loss_list.append(epoch_loss)
            training_dice_list.append(epoch_acc)

        print("Best test Balanced acc: {:4f}".format(best_acc))
        print("----- Training Loss ---------")
        print(training_loss_list)
        print("------Validation Accuracy ------")
        print(training_dice_list)
        # load best model weights
        self.benchmark_model.load_state_dict(best_model_wts)
        return self.benchmark_model

    # def convert_dataloader_to_dataloader_w_id(
    #     self, list_of_dataloaders: DataLoader, batch_sizes: list = None
    # ):

    #     # handle if batch_sizes is None
    #     if not batch_sizes:
    #         batch_sizes = [None] * len(list_of_dataloaders)

    #     dataloaders_w_id = []
    #     for idx, dataloader in enumerate(list_of_dataloaders):
    #         ds = dataloader.dataset
    #         ds_w_id = DatasetWithIDWrapper(ds)

    #         if self.dataset == "FedCamelyon16":
    #             dataloader_w_id = DataLoader(
    #                 ds_w_id,
    #                 batch_size=1 if batch_sizes[idx] is None else batch_sizes[idx],
    #                 shuffle=dataloader.shuffle if "shuffle" in dataloader else False,
    #                 collate_fn=collate_fn,
    #             )
    #         else:
    #             dataloader_w_id = DataLoader(
    #                 ds_w_id,
    #                 batch_size=1 if batch_sizes[idx] is None else batch_sizes[idx],
    #                 shuffle=dataloader.shuffle if "shuffle" in dataloader else False,
    #             )
    #         dataloaders_w_id.append(dataloader_w_id)
    #     return dataloaders_w_id

    def dataloaders_to_custom_batchsize(
        self, list_of_dataloaders: DataLoader, batch_sizes: list = None
    ):
        new_dataloaders = []
        for idx, dataloader in enumerate(list_of_dataloaders):
            ds = dataloader.dataset
            if self.dataset == "FedCamelyon16":
                new_dataloader = DataLoader(
                    ds,
                    batch_sizes[idx],
                    shuffle=dataloader.shuffle if "shuffle" in dataloader else False,
                    collate_fn=collate_fn,
                )
            else:
                new_dataloader = DataLoader(
                    ds,
                    batch_sizes[idx],
                    shuffle=dataloader.shuffle if "shuffle" in dataloader else False,
                )
            new_dataloaders.append(new_dataloader)
        print(f"Converted dataloaders to custom batch size {batch_sizes}")
        return new_dataloaders

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
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for inputs, labels, hash_id in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                loss = self.args["loss"].forward(outputs, labels)
                sample_losses.append({hash_id[0]: loss.item()})

        return sample_losses

    def select_train_samples(self, benchmark_test_samples_loss, train_samples_loss):
        """
        Selects a subset of training samples based on loss values relative to a benchmark dataset.

        This function implements the data selection process proposed by Tuor et al. in
        "Overcoming Noisy and Irrelevant Data in Federated Learning," ICPR 2021.
        It aims to retain training samples with loss values similar to those in a benchmark
        validation dataset, thereby enhancing the quality of the training data.

        Parameters
        ----------
        benchmark_test_samples_loss : list of dicts
            Loss values for samples in the benchmark validation dataset.

        train_samples_loss : list of dicts
            Loss values for samples in the training dataset.

        Returns
        -------
        kept_train_samples_loss : list of dicts
            A subset of training samples with the lowest loss values, selected based on
            the Kolmogorov-Smirnov (KS) distance from the benchmark dataset.

        Notes
        -----
        The function computes the cumulative distribution function (CDF) for both datasets,
        calculates the maximum KS distance, and retains samples with loss values below a
        determined threshold to minimize the inclusion of noisy or irrelevant data.
        """

        # Adaptations for our cross-silo setting:
        # - cross-silo w/ less data per client -> adapt threshold (lambda) steps size to 1
        # - cross-silo w/ less data per client -> increase resolution of CDFs => numbins=10000

        max_each_cut = []
        steps = np.arange(1, len(train_samples_loss), 1)

        # F_v: cdf of benchmark validation ds
        benchmark_test_samples_loss_values = [
            list(d.values())[0] for d in benchmark_test_samples_loss
        ]
        cdf = stats.cumfreq(
            benchmark_test_samples_loss_values, numbins=10000, defaultreallimits=(0, 30)
        )
        train_samples_loss_values = [list(d.values())[0] for d in train_samples_loss]
        for t in steps:
            # F_p: cdf of training ds
            cdf2 = stats.cumfreq(
                train_samples_loss_values[0:t], numbins=10000, defaultreallimits=(0, 30)
            )
            # compute KS-distance G
            difference = abs(
                cdf.cumcount / cdf.cumcount[-1] - cdf2.cumcount / cdf2.cumcount[-1]
            )
            max_each_cut.append(max(difference))

        # keep n_samples_to_keep which's training ds loss value on benchmark model w_val are <= threshold (= np.argmin(max_each_cut))
        n_sample_to_keep = steps[np.argmin(max_each_cut)]
        print(
            f"KS Loss data selection is keeping {n_sample_to_keep} from {len(train_samples_loss)} samples in the distributed clients!"
        )
        kept_train_samples_loss = train_samples_loss[:n_sample_to_keep]

        return kept_train_samples_loss

    def filter_kept_samples_per_client(
        self, kept_train_samples_per_client, train_dataloaders
    ):
        # Step 1: Collect kept hash_ids per client
        kept_hash_ids_per_client = [
            {list(sample.keys())[0] for sample in client_samples}
            for client_samples in kept_train_samples_per_client
        ]

        # Step 2: Filter each dataloader's dataset based on the kept hash_ids
        filtered_dataloaders = []
        for client_index, dataloader in enumerate(train_dataloaders):
            # Get the dataset from the current dataloader
            dataset = dataloader.dataset

            # Identify samples with matching hash_ids and collect their indices
            indices_to_keep = [
                i
                for i, sample in enumerate(dataset)
                if sample[2]
                in kept_hash_ids_per_client[
                    client_index
                ]  # assuming the hash_id is [2] element of the sample
            ]

            if len(indices_to_keep) == 0:
                print(f"Client {client_index} has no samples to keep!")
                filtered_dataloaders.append([])
                continue

            kept_dataset = Subset(dataloader.dataset, indices_to_keep)

            if self.dataset == "FedCamelyon16":
                filtered_dataloader = DataLoader(
                    kept_dataset,
                    batch_size=dataloader.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                )
            else:
                filtered_dataloader = DataLoader(
                    kept_dataset,
                    batch_size=dataloader.batch_size,
                    shuffle=True,
                )

            filtered_dataloaders.append(filtered_dataloader)
            print(
                f"Client {client_index} keeps {len(filtered_dataloader.dataset)} samples!"
            )
        return filtered_dataloaders

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
        trained_benchmark_model = self.train_benchmark_model(
            benchmark_train_loader, benchmark_test_loader
        )

        # Step 4: Convert dataloader to dataloaders with custom batch_size
        benchmark_test_loader_b1 = self.dataloaders_to_custom_batchsize(
            [benchmark_test_loader], batch_sizes=[1]
        )[0]
        train_dataloaders_b1 = self.dataloaders_to_custom_batchsize(
            train_dataloaders, batch_sizes=[1 for i in range(0, len(train_dataloaders))]
        )

        # Step 5: Evaluate on benchmark test set
        benchmark_test_sample_benchmark_loss = self.compute_sample_losses(
            trained_benchmark_model, benchmark_test_loader_b1
        )
        print("Sample losses on benchmark test set computed!")

        # Step 6: Evaluate on each client train datasets
        train_sample_benchmark_loss_per_client = []
        for idx, train_dataloader_b1 in enumerate(train_dataloaders_b1):
            train_sample_benchmark_loss_per_client.append(
                self.compute_sample_losses(trained_benchmark_model, train_dataloader_b1)
            )
            print(f"Sample losses on client {idx} train set computed!")

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

        print(
            f"benchmark_test_sample_benchmark_loss values: {benchmark_test_sample_benchmark_loss}"
        )
        print(
            f"sorted_train_sample_benchmark_loss_all_clients values: {sorted_train_sample_benchmark_loss_all_clients}"
        )
        # Step 8: Select samples to keep based on sorted loss values
        kept_train_sample_benchmark_loss_all_clients = self.select_train_samples(
            benchmark_test_sample_benchmark_loss,
            sorted_train_sample_benchmark_loss_all_clients,
        )

        # Step 9: Retrieve client reference back per kept data sample
        # Extract hash IDs of kept samples
        kept_hash_ids = {
            list(sample.keys())[0]
            for sample in kept_train_sample_benchmark_loss_all_clients
        }
        # Filter each client's samples based on kept_hash_ids
        kept_train_samples_per_client = [
            [
                sample
                for sample in client_samples
                if list(sample.keys())[0] in kept_hash_ids
            ]
            for client_samples in train_sample_benchmark_loss_per_client
        ]

        # Step 10: Compose from kept data samples per client the clients data loaders
        filtered_dataloaders = self.filter_kept_samples_per_client(
            kept_train_samples_per_client, train_dataloaders
        )

        return filtered_dataloaders
