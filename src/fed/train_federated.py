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
        # 2 lines of code to change to switch to another dataset
        from flamby.datasets.fed_heart_disease import (
            BATCH_SIZE,
            LR,
            NUM_EPOCHS_POOLED,
            Baseline,
            BaselineLoss,
            metric,
            NUM_CLIENTS,
            get_nb_max_rounds
        )
        from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset

    elif ds_name == "FedCamelyon16":
        print(0)
        #TODO get fedcamelyon16 dataset
    # TODO implement more datasets

    else:
        # Raise an exception if an unsupported dataset name is provided
        raise ValueError(f"Dataset '{ds_name}' is not supported. Please provide a valid dataset name.")


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

    #TODO: implement more strategies

    else:
        # Raise an exception if an unsupported strategy name is provided
        raise ValueError(f"Federated aggregation strategy '{agg_strategy_name}' is not supported. Please provide a valid strategy name.")



def train_federated(args):

    # import dataset and aggregation strategy and set their params
    import_dataset(args.dataset)
    import_n_set_agg_strategy(args.agg_strategy)
    print(f"Dataset set: {args.dataset}")
    print(f"Federated aggregation strategy set: {args.agg_strategy}")

    # We loop on all the clients of the distributed dataset and instantiate associated data loaders
    train_dataloaders = [
                torch.utils.data.DataLoader(
                    FedDataset(center = i, train = True, pooled = False),
                    batch_size = BATCH_SIZE,
                    shuffle = True,
                    num_workers = 0
                )
                for i in range(NUM_CLIENTS)
            ]

    lossfunc = BaselineLoss()
    m = Baseline()

    # Federated Learning loop
    # 2nd line of code to change to switch to another strategy (feed the FL strategy the right HPs)
    args = {
                "training_dataloaders": train_dataloaders,
                "model": m,
                "loss": lossfunc,
                "optimizer_class": torch.optim.SGD,
                "learning_rate": LR / 10.0,
                "num_updates": 100,
    # This helper function returns the number of rounds necessary to perform approximately as many
    # epochs on each local dataset as with the pooled training
                "nrounds": get_nb_max_rounds(100),
            }
    s = strat(**args)

    # full training is performed according to and in defined strategy
    m = s.run()[0]

    # Evaluation
    # We only instantiate one test set in this particular case: the pooled one
    test_dataloaders = [
                torch.utils.data.DataLoader(
                    FedDataset(train = False, pooled = True),
                    batch_size = BATCH_SIZE,
                    shuffle = False,
                    num_workers = 0,
                )
            ]
    dict_cindex = evaluate_model_on_tests(m, test_dataloaders, metric)
    print(dict_cindex)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, default='FedHeartDisease',
        help='[REQUIRED] Name of the Dataset that should be used. Default is FedHeartDisease.')
    parser.add_argument('--agg_strategy', type=str, required=True, default='FedAvg',
        help='[REQUIRED] Name of the federated aggregation strategy that should be used. Default is FedAvg.')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    train_federated(args)