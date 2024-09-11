import dask.array as da
from xgboost.dask import DaskXGBClassifier
from lightgbm import DaskLGBMClassifier
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
import numpy as np
from dask_ml.model_selection import GridSearchCV as dgsc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm


def load_cluster(device: str):
    if device == 'cuda':
        cluster = LocalCUDACluster()
        client = Client(cluster)
        print(device)
    else:
        cluster = LocalCluster()
        client = cluster.get_client()
        print(device)

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


def dask_xgboost(client: Client,
                 device: str):

    estimator = DaskXGBClassifier(objective='multi:softmax',
                                  num_class=4,
                                  tree_method='hist',
                                  device=device,
                                  random_state=1337)
    estimator.client = client
    return estimator


def dask_lgbm(client: Client) -> DaskLGBMClassifier:

    estimator = DaskLGBMClassifier(objective='multiclass',
                                   random_state=1337,
                                   client=client)
    return estimator


def gridsearch(x_train: da,
               x_test: da,
               y_train: da,
               y_test: da,
               estimator,
               ParamGrid: dict,
               client):

    GridSearch = tqdm(dgsc(estimator,
                      param_grid=ParamGrid,
                      scoring='accuracy',
                      cv=5,
                      verbose=3,
                      client=client))

    GridSearch.fit(x_train, y_train)
    y_pred = GridSearch.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nAccuracy: {accuracy: .7f}')

    precision, recall, F1Score, _ = precision_recall_fscore_support(y_test,
                                                                    y_pred)

    for i, v in enumerate(['P1', 'P2', 'P3', 'P4']):
        print(f"Class {v}")
        print(f"Precision: {precision[i]: .7f}")
        print(f"Recall: {recall[i]: .7f}")
        print(f"F1 Score: {F1Score[i]: .7f}")
    return GridSearch
