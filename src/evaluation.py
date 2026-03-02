from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def evaluate_binary(y_true, y_pred, y_prob=None):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    return metrics


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def evaluate_multiclass(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted")
    }

    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return metrics, report, cm

