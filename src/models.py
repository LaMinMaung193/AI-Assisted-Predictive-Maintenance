#Binary Class

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_logistic():
    return LogisticRegression(max_iter=1000)


def get_decision_tree():
    return DecisionTreeClassifier(random_state=42)


def get_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )


def get_svm():
    return SVC(kernel="rbf", probability=True)

# src/models.py
#Multiclass


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_logistic_multi():
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

def get_rf_multi():
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )


def get_dt_multi():
    return DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    )


def get_svm_multi():
    return SVC(
        probability=True,
        class_weight="balanced"
    )