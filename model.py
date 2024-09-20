from base_files.preprocessing.cleaning import filter_nan, join_df
from base_files.preprocessing.loading import read_file, col_casting
from base_files.preprocessing.transform import (suitable_cat_col,
                                                treating_every_col)
from base_files.models.catboost_classifier import cb_classifier
from base_files.models.lgb_classifier import lgbm
from base_files.models.xgboost_classifier import xgboost_initializer
from base_files.models.tree_classifier import rf_classifer, dt_classifer
from base_files.models.save import model_save
from base_files.utils.check_score import accuracy
from base_files.utils.grid_search import filter_model
from base_files.utils.model_prep import split_data
from argparse import ArgumentParser
import GPUtil
import warnings
warnings.filterwarnings('ignore')


def train(client,
          df1_path: str,
          df2_path: str):

    # Reading the data
    df1, df2 = read_file(df1_path, df2_path)
    # print(df1.shape, df2.shape)

    # Reducing the column size
    df1, df2 = col_casting(df1, df2)
    # print(df1[:5])
    # print(df2[:5])

    # Filtering columns on the basis of 10000 NaN's
    df1, df2 = filter_nan(df1, df2)
    # Joining DataFrame
    df = join_df(df1, df2)
    # print(df.shape)
    # print(df.null_count().sum_horizontal())

# checking contingency of categorical variable with our target variable
    df = suitable_cat_col(df)
    # print(df.shape)
    df = treating_every_col(df)
    # print(df.shape)

    # Splitting data into features and labels
    y = df[:, 'Approved_Flag']
    x = df.drop('Approved_Flag')

    # Splitting data
    (x_train,
     x_test,
     y_train,
     y_test) = split_data(TrainData=x,
                          LabelData=y)
    # Dictionary to calculate model accuracy
    ModelAccuracy = {}

    # Random Forest model
    print("RANDOM FOREST")
    RandomForest = rf_classifer(NumEsti=200,
                                RandomState=1337)
    RandomForest.fit(x_train, y_train)
    RFAccuracy = accuracy(RandomForest,
                          x_test,
                          y_test)
    ModelAccuracy['random_forest'] = RFAccuracy * 100
    del RandomForest
    del RFAccuracy

    # Decision Tree model
    print("DECISION TREE")
    DecisionTree = dt_classifer(RandomState=1337)
    DecisionTree.fit(x_train, y_train)
    DTAccuracy = accuracy(DecisionTree,
                          x_test,
                          y_test)
    ModelAccuracy['decision_tree'] = DTAccuracy * 100
    del DecisionTree
    del DTAccuracy

    # Decision Tree model
    print("XGBOOST CLASSIFIER")
    if client is not None:
        (da_x_train,
         da_x_test,
         da_y_train,
         da_y_test) = data_converter_dask(x_train,
                                          x_test,
                                          y_train,
                                          y_test)
    else:
        (da_x_train,
         da_x_test,
         da_y_train,
         da_y_test) = (x_train,
                       x_test,
                       y_train,
                       y_test)

    XgbClassifier = xgboost_initializer(client=client,
                                        device=device,
                                        params=None,
                                        NumClass=4,
                                        RandomState=1337)
    XgbClassifier.fit(da_x_train, da_y_train)
    XGBAccuracy = accuracy(XgbClassifier,
                           da_x_test,
                           da_y_test)
    ModelAccuracy['xgboost'] = XGBAccuracy * 100
    del XgbClassifier
    del XGBAccuracy

    # CatBoost Classifier model
    print("CATBOOST CLASSIFIER")
    CbClassifier = cb_classifier(device=device,
                                 params=None,
                                 RandomState=1337,
                                 verbose=False)
    CbClassifier.fit(x_train, y_train)
    CBAccuracy = accuracy(CbClassifier,
                          x_test,
                          y_test)
    ModelAccuracy['catboost'] = CBAccuracy * 100
    del CbClassifier
    del CBAccuracy

    # CatBoost Classifier model
    print("LIGHTGBM CLASSIFIER")
    LgbmClassifier = lgbm(device=device,
                          params=None,
                          RandomState=1337,
                          Verbosity=-1)
    LgbmClassifier.fit(x_train, y_train)
    LGBMAccuracy = accuracy(LgbmClassifier,
                            x_test,
                            y_test)
    ModelAccuracy['lightgbm'] = LGBMAccuracy * 100
    del LgbmClassifier
    del LGBMAccuracy

    # Filtering models for accuracy
    FilterAcc = max(ModelAccuracy.values()) - 0.2
    FilteredModel = []
    for x in ModelAccuracy.keys():
        if ModelAccuracy[x] > FilterAcc:
            FilteredModel.append(x)
    del FilterAcc

    # print(FilteredModel)
    # Applying gridsearch
    if 'lightgbm' in FilteredModel:
        Model = filter_model('lightgbm',
                             x_train,
                             x_test,
                             y_train,
                             y_test,
                             device=device,
                             client=client)
        FilteredModel.remove('lightgbm')

    BestModelName = 'lightgbm'
    BestModel = Model[0]
    BestScore = Model[1]
    for ModelName in FilteredModel:
        if ModelName == 'xgboost':
            Model = filter_model(ModelName,
                                 da_x_train,
                                 da_x_test,
                                 da_y_train,
                                 da_y_test,
                                 device=device,
                                 client=client)
        else:
            Model = filter_model(ModelName,
                                 x_train,
                                 x_test,
                                 y_train,
                                 y_test,
                                 device=device,
                                 client=client)

        if Model[1] > BestScore:
            BestModelName = x
            BestModel = Model[0]
            BestScore = Model[1]

    if client is not None:
        client.close()
    return BestModelName, BestModel, x_train, y_train


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    TotalGPUs = len(GPUtil.getAvailable())
    device = 'cuda'
    cluster, client = None, None
    if TotalGPUs > 1:
        from base_files.utils.clusters import load_cluster, data_converter_dask
        cluster, client = load_cluster()
        print('Using GPU clusters')
    if TotalGPUs == 1:
        print('Using single GPU')
    else:
        device = 'cpu'

    # Argument Parsing
    parser = ArgumentParser()
    parser.add_argument('--path', dest='Paths', action='append')
    paths = parser.parse_args()

    if paths.Paths[0] == "N":
        df1_path = 'case_study1.xlsx'
        df2_path = 'case_study2.xlsx'
    else:
        df1_path = paths.Paths[0]
        df2_path = paths.Paths[1]

    (BestModelName,
     BestModel,
     x_train,
     y_train) = train(client,
                      df1_path,
                      df2_path)

    if cluster is not None:
        cluster.close()

    a = model_save(estimator=BestModel,
                   ModelName=BestModelName,
                   x_train=x_train,
                   y_train=y_train,
                   RandomState=1337)
