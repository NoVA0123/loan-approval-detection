from catboost import CatBoostClassifier


def cb_classifier(device: str,
                  params: dict | None = None,
                  RandomState: int = 1337,
                  verbose: int | bool = False):

    if device == 'cuda':
        TaskType = 'GPU'
    else:
        TaskType = 'CPU'

    CBClassifier = CatBoostClassifier(params,
                                      objective='MultiClass',
                                      task_type=TaskType,
                                      devices='0',
                                      random_state=1337,
                                      verbose=False)

    return CBClassifier
