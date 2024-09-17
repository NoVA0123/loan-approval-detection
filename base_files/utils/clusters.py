from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import numpy as np
import dask.array as da


def load_cluster():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    return cluster, client


def data_converter_dask(x_train: np.array,
                        x_test: np.array,
                        y_train: np.array,
                        y_test: np.array) -> tuple:

    x_train = da.from_array(x_train)
    x_test = da.from_array(x_test)
    y_train = da.from_array(y_train)
    y_test = da.from_array(y_test)

    return x_train, x_test, y_train, y_test
