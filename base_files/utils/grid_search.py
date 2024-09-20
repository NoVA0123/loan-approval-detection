from base_files.utils.check_score import accuracy
from base_files.models.catboost_classifier import cb_classifier
from base_files.models.lgb_classifier import lgbm
from base_files.models.xgboost_classifier import xgboost_initializer
from base_files.models.tree_classifier import rf_classifer, dt_classifer
from sklearn.model_selection import GridSearchCV


def gridsearch(x_train,
               x_test,
               y_train,
               y_test,
               estimator,
               ParamGrid: dict,
               verbose: int = 0):

    GridSearch = GridSearchCV(estimator,
                              param_grid=ParamGrid,
                              scoring='accuracy',
                              cv=3,
                              verbose=verbose)

    GridSearch.fit(x_train, y_train)

    FinalScore = accuracy(model=GridSearch,
                          x_test=x_test,
                          y_test=y_test)

    return GridSearch, FinalScore


def filter_model(ModelName: str,
                 x_train,
                 x_test,
                 y_train,
                 y_test,
                 device: str,
                 client):
    match ModelName:
        case 'random_forest':
            print("\nGRID SEARCH ON RANDOM FOREST")
            param_grid = {
                    'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_depth': [3, 5, 8, 10],
                    'ccp_alpha': [1, 10, 100],
                    'n_estimators': [10, 50, 100]
                    }
            estimator = rf_classifer(NumEsti=200,
                                     RandomState=1337)
            GridSearch, Accuracy = gridsearch(x_train,
                                              x_test,
                                              y_train,
                                              y_test,
                                              estimator,
                                              param_grid)

        case 'decision_tree':
            print("\nGRID SEARCH ON DECISION TREE")
            param_grid = {
                    'max_depth': [3, 5, 8, 10],
                    'ccp_alpha': [1, 10, 100],
                    }
            estimator = dt_classifer(RandomState=1337)
            GridSearch, Accuracy = gridsearch(x_train,
                                              x_test,
                                              y_train,
                                              y_test,
                                              estimator,
                                              param_grid)

        case 'xgboost':
            print("\nGRID SEARCH ON XGBOOST")
            param_grid = {
                    'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'learning_rate': [0.001, 0.01, 0.1, 1],
                    'max_depth': [3, 5, 8, 10],
                    'alpha': [1, 10, 100],
                    'n_estimators': [10, 50, 100]
                    }
            estimator = xgboost_initializer(client=client,
                                            device=device,
                                            params=None,
                                            NumClass=4,
                                            RandomState=1337)
            GridSearch, Accuracy = gridsearch(x_train,
                                              x_test,
                                              y_train,
                                              y_test,
                                              estimator,
                                              param_grid)

        case 'lightgbm':
            print("\nGRID SEARCH ON LIGHTGBM")
            param_grid = {
                    'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'learning_rate': [0.001, 0.01, 0.1, 1],
                    'max_depth': [3, 5, 8, 10],
                    'reg_alpha': [1, 10, 100],
                    'n_estimators': [10, 50, 100]
                    }
            estimator = lgbm(device=device,
                             params=None,
                             RandomState=1337,
                             Verbosity=-1)
            GridSearch, Accuracy = gridsearch(x_train,
                                              x_test,
                                              y_train,
                                              y_test,
                                              estimator,
                                              param_grid,
                                              verbose=0)
            print(GridSearch, Accuracy)

        case 'catboost':
            print("\nGRID SEARCH ON CATBOOST")
            param_grid = {
                    'colsample_bylevel': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'learning_rate': [0.001, 0.01, 0.1, 1],
                    'max_depth': [3, 5, 8, 10],
                    'reg_lambda': [1, 10, 100],
                    'n_estimators': [10, 50, 100]
                    }
            estimator = cb_classifier(device=device,
                                      params=None,
                                      RandomState=1337,
                                      verbose=False)
            GridSearch, Accuracy = gridsearch(x_train,
                                              x_test,
                                              y_train,
                                              y_test,
                                              estimator,
                                              param_grid)
    return [GridSearch, Accuracy]
