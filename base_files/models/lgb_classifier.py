from lightgbm import LGBMClassifier


def lgbm(device: str,
         params: dict | None = None,
         RandomState: int = 1337,
         Verbosity: int = -1) -> LGBMClassifier:

    device = 'cpu'
    estimator = LGBMClassifier(params,
                               objective='multiclass',
                               device=device,
                               random_state=RandomState,
                               verbosity=Verbosity)
    return estimator
