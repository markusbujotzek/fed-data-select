
from FLamby.flamby.datasets.fed_tcga_brca import TcgaBrcaRaw, FedTcgaBrca

def load_datasets():
    # Raw dataset
    mydataset_raw = TcgaBrcaRaw()

    # Pooled test dataset
    mydataset_pooled = FedTcgaBrca(train=False, pooled=True)

    # Center 2 train dataset
    mydataset_local2= FedTcgaBrca(center=2, train=True, pooled=False)

    return mydataset_raw, mydataset_pooled, mydataset_local2