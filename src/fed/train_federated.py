import torch
import argparse
from flamby.utils import evaluate_model_on_tests


def import_dataset(ds_name):
    """
    Import dataset and dataset-related parameters.

    params:
    * ds_name (str): Name of the dataset. Supported datasets include "FedHeartDisease", "FedCamelyon16".
    """

    global BATCH_SIZE, LR, NUM_EPOCHS_POOLED, Baseline, BaselineLoss, metric, NUM_CLIENTS, get_nb_max_rounds, FedDataset

    if ds_name == "FedHeartDisease":
        from flamby.datasets.fed_heart_disease import (
            BATCH_SIZE,
            LR,
            NUM_EPOCHS_POOLED,
            Baseline,
            BaselineLoss,
            metric,
            NUM_CLIENTS,
            get_nb_max_rounds,
        )
        from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset

    elif ds_name == "FedCamelyon16":
        global collate_fn
        from flamby.datasets.fed_camelyon16 import (
            BATCH_SIZE,
            LR,
            NUM_EPOCHS_POOLED,
            Baseline,
            BaselineLoss,
            metric,
            NUM_CLIENTS,
            get_nb_max_rounds,
        )
        from flamby.datasets.fed_camelyon16 import FedCamelyon16 as FedDataset

        # dataset requires padding implemented in collate_fn
        from flamby.datasets.fed_camelyon16 import collate_fn
    elif ds_name == "FedKits19":
        from flamby.datasets.fed_kits19 import (
            BATCH_SIZE,
            LR,
            NUM_EPOCHS_POOLED,
            Baseline,
            BaselineLoss,
            metric,
            NUM_CLIENTS,
            get_nb_max_rounds,
        )
        from flamby.datasets.fed_kits19 import FedKits19 as FedDataset

    # TODO implement more datasets

    else:
        # Raise an exception if an unsupported dataset name is provided
        raise ValueError(
            f"Dataset '{ds_name}' is not supported. Please provide a valid dataset name."
        )


def import_n_set_agg_strategy(agg_strategy_name):
    """
    Import federated aggregation strategy and its parameters.

    params:
    * agg_strategy_name (str): Name of the federated aggregation strategy. Supported strategies include "FedAvg".
    """
    global strat

    if agg_strategy_name == "FedAvg":
        # 1st line of code to change to switch to another strategy
        from flamby.strategies.fed_avg import FedAvg as strat

        # TODO import aggregation strategy related hparams

    # TODO: implement more strategies

    else:
        # Raise an exception if an unsupported strategy name is provided
        raise ValueError(
            f"Federated aggregation strategy '{agg_strategy_name}' is not supported. Please provide a valid strategy name."
        )


def import_data_selection_method(data_selection_method):
    global data_select_method

    if data_selection_method == "KSLoss":
        from data_selection_methods.KSLoss.ksloss import KSLoss as data_select_method

    elif not data_selection_method:
        print(
            "No dataselection method defined. You seem to enjoy wasting model performance, computational costs and energy!"
        )

    else:
        # Raise an exception if sth went wrong
        raise ValueError(
            f"Data selection method '{data_selection_method}' is not supported. Please provide a valid method name."
        )


def train_federated(arparser_argsgs):

    # import dataset
    import_dataset(parser_args.dataset)
    print(f"Dataset set: {parser_args.dataset}")
    # import aggregation strategy and set their params
    import_n_set_agg_strategy(parser_args.agg_strategy)
    print(f"Federated aggregation strategy set: {parser_args.agg_strategy}")
    # import data selection method
    import_data_selection_method(parser_args.data_selection)
    print(f"Data selection method set: {parser_args.data_selection}")

    # instantiate data loaders
    if parser_args.dataset == "FedCamelyon16":
        train_dataloaders = [
            torch.utils.data.DataLoader(
                FedDataset(center=i, train=True, pooled=False),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=12,
                collate_fn=collate_fn,  # Use collate_fn only for FedCamelyon16
            )
            for i in range(NUM_CLIENTS)
        ]
    else:
        train_dataloaders = [
            torch.utils.data.DataLoader(
                FedDataset(center=i, train=True, pooled=False),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=12,
            )
            for i in range(NUM_CLIENTS)
        ]

    # set loss and model
    lossfunc = BaselineLoss()
    m = Baseline()

    # set general arguments for fl training
    general_args = {
        "training_dataloaders": train_dataloaders,
        "model": m,
        "loss": lossfunc,
        "optimizer_class": torch.optim.SGD,
        "learning_rate": LR,  #  / 10.0,
        "num_updates": 10,  # batches per local epoch
        # This helper function returns the number of rounds necessary to perform approximately as many
        # epochs on each local dataset as with the pooled training
        "nrounds": get_nb_max_rounds(500),  # fl communication round
        "log": True,
        "log_period": 5,
    }

    # manipulate training data loaders accoring to data selection method
    if parser_args.data_selection:
        if parser_args.data_selection == "KSLoss":
            ksloss_data_select = data_select_method(
                benchmark_model=m,
                loss_fn=lossfunc,
                metric=metric,
                batch_size=BATCH_SIZE,
                args={
                    **general_args,
                    "dataset": parser_args.dataset,
                    "benchmark_ds_percentage": parser_args.benchmark_ds_percentage,
                    "benchmark_train_ratio": parser_args.benchmark_train_ratio,
                    "num_epochs_benchmark_model_training": parser_args.num_epochs_benchmark_model_training,
                },
            )
            train_dataloaders = ksloss_data_select.ksloss_data_selection(
                train_dataloaders
            )

    s = strat(**general_args)

    # full federated training is performed according to and in defined strategy
    m = s.run()[0]

    # Evaluation
    # We only instantiate one test set in this particular case: the pooled one
    if parser_args.dataset == "FedCamelyon16":
        test_dataloaders = [
            torch.utils.data.DataLoader(
                FedDataset(train=False, pooled=True),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,  # Use collate_fn only for FedCamelyon16
            )
        ]
    else:
        test_dataloaders = [
            torch.utils.data.DataLoader(
                FedDataset(train=False, pooled=True),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
        ]
    dict_cindex = evaluate_model_on_tests(m, test_dataloaders, metric)
    print(dict_cindex)


def get_args():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="FedHeartDisease",
        help="[REQUIRED] Name of the Dataset that should be used. Default is FedHeartDisease.",
    )
    parser.add_argument(
        "--agg_strategy",
        type=str,
        required=True,
        default="FedAvg",
        help="[REQUIRED] Name of the federated aggregation strategy that should be used. Default is FedAvg.",
    )
    parser.add_argument(
        "--data_selection",
        default=None,
        help="[OPTIONAL] Name of the data selection strategy that should be used. Default is None.",
    )

    # KSLoss specific arguments
    parser.add_argument(
        "--benchmark_ds_percentage",
        type=float,
        default=0.05,
        help="[OPTIONAL] Percentage of client's data that is given to the benchmark dataset. Default is 0.05.",
    )
    parser.add_argument(
        "--benchmark_train_ratio",
        type=float,
        default=0.8,
        help="[OPTIONAL] Ratio of the benchmark dataset that is used for training. Default is 0.8.",
    )
    parser.add_argument(
        "--num_epochs_benchmark_model_training",
        type=int,
        default=20,
        help="[OPTIONAL] Number of epochs the benchmark model is trained for. Default is 20.",
    )

    parser_args = parser.parse_args()

    return parser_args


if __name__ == "__main__":
    parser_args = get_args()
    train_federated(parser_args)
