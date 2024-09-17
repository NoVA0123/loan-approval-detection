from typing import Any
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support)


def accuracy(model: Any,
             x_test,
             y_test) -> float:

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nAccuracy: {accuracy: .7f}')

    precision, recall, F1Score, _ = precision_recall_fscore_support(y_test,
                                                                    y_pred)

    for i, v in enumerate(['P1', 'P2', 'P3', 'P4']):
        print(f"Class {v}")
        print(f"Precision: {precision[i]: .7f}")
        print(f"Recall: {recall[i]: .7f}")
        print(f"F1 Score: {F1Score[i]: .7f}")

    return accuracy
