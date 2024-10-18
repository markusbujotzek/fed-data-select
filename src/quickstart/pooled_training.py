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
    Optimizer,
)
from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset

# Instantiation of local train set (and data loader)), baseline loss function, baseline model, default optimizer
train_dataset = FedDataset(center=0, train=True, pooled=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
lossfunc = BaselineLoss()
model = Baseline()
optimizer = Optimizer(model.parameters(), lr=LR)

# Traditional pytorch training loop
for epoch in range(0, NUM_EPOCHS_POOLED):
    for idx, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(X)
        loss = lossfunc(outputs, y)
        loss.backward()
        optimizer.step()

# Evaluation
# Instantiation of a list of the local test sets
test_dataloaders = [
            torch.utils.data.DataLoader(
                FedDataset(center=i, train=False, pooled=False),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
            for i in range(NUM_CLIENTS)
        ]
# Function performing the evaluation
dict_cindex = evaluate_model_on_tests(model, test_dataloaders, metric)
print(dict_cindex)
