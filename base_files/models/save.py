from base_files.models.catboost_classifier import cb_classifier
from base_files.models.lgb_classifier import lgbm
from base_files.models.xgboost_classifier import xgboost_initializer
import pickle


def model_save(estimator,
               ModelName: str,
               device: str,
               x_train,
               y_train,
               RandomState: int = 1337):

    match ModelName:
        case 'random_forest':
            BestEsti = estimator.best_estimator_
            pickle.dump(BestEsti, open('rf_loan_fraud.pkl', 'wb'))

        case 'decision_tree':
            BestEsti = estimator.best_estimator_()
            pickle.dump(BestEsti, open('dt_loan_fraud.pkl', 'wb'))

        case 'xgboost':
            BestParams = estimator.best_params_
            print(BestParams)
            BestEsti = xgboost_initializer(client=None,
                                           device=device,
                                           params=BestParams,
                                           NumClass=4,
                                           RandomState=RandomState)
            BestEsti.fit(x_train, y_train)
            BestEsti.save_model("xgb_loan_fraud.json")

        case 'ligthgbm':
            BestParams = estimator.best_params_
            print(BestParams)
            BestEsti = lgbm(device=device,
                            params=BestParams,
                            RandomState=RandomState,
                            Verbosity=-1)
            BestEsti.fit(x_train, y_train)
            BestEsti.booster_.save_model('lgbm_loan_fraud.txt')

        case 'catboost':
            BestParams = estimator.best_params_
            print(BestParams)
            BestEsti = cb_classifier(device=device,
                                     params=BestParams,
                                     RandomState=RandomState,
                                     verbose=False)
            BestEsti.fit(x_train, y_train)
            BestEsti.save_model('cb_loan_fraud',
                                format='cbm')

    return f'{ModelName} has been saved'
