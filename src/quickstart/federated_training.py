import torch
from flamby.utils import evaluate_model_on_tests

# 2 lines of code to change to switch to another dataset
from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    get_nb_max_rounds
)
from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset

# 1st line of code to change to switch to another strategy
from flamby.strategies.fed_avg import FedAvg as strat

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
