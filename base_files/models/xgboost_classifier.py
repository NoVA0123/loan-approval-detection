import xgboost as xgb
from xgboost.dask import DaskXGBClassifier
from dask.distributed import Client


def xgboost_initializer(client: Client | None,
                        device: str,
                        params: dict | None = None,
                        NumClass: int = 4,
                        RandomState: int = 1337):

    if client is not None:
        estimator = DaskXGBClassifier(params,
                                      objective='multi:softmax',
                                      num_class=NumClass,
                                      tree_method='hist',
                                      device=device,
                                      random_state=RandomState)
        estimator.client = client

    else:
        estimator = xgb.XGBClassifier(params,
                                      objective='multi:softmax',
                                      num_class=NumClass,
                                      tree_method='hist',
                                      device=device,
                                      random_state=RandomState)
    return estimator
