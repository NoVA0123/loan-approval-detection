from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def rf_classifer(NumEsti: int = 200,
                 RandomState: int = 1337):

    RfClassifier = RandomForestClassifier(n_estimators=NumEsti,
                                          random_state=RandomState)

    return RfClassifier


def dt_classifer(RandomState: int = 1337):

    RfClassifier = DecisionTreeClassifier(random_state=RandomState)

    return RfClassifier
