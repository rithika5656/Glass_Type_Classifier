from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from typing import Any

from . import config


def create_knn_model(n_neighbors: int = None) -> KNeighborsClassifier:

    """Create KNN classifier."""
    if n_neighbors is None:
        n_neighbors = config.KNN_N_NEIGHBORS
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def create_svm_model(kernel: str = None, random_state: int = None) -> SVC:

    """Create SVM classifier."""
    if kernel is None:
        kernel = config.SVM_KERNEL
    if random_state is None:
        random_state = config.RANDOM_STATE
    return SVC(kernel=kernel, random_state=random_state)

def create_rf_model(n_estimators: int = None, random_state: int = None) -> RandomForestClassifier:
    """Create Random Forest classifier."""
    if n_estimators is None:
        n_estimators = config.RF_N_ESTIMATORS
    if random_state is None:
        random_state = config.RANDOM_STATE
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

def create_gb_model(learning_rate: float = None, n_estimators: int = None, random_state: int = None) -> GradientBoostingClassifier:
    """Create Gradient Boosting classifier."""
    if learning_rate is None:
        learning_rate = config.GB_LEARNING_RATE
    if n_estimators is None:
        n_estimators = config.GB_N_ESTIMATORS
    if random_state is None:
        random_state = config.RANDOM_STATE
    return GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=random_state)

def create_lr_model(max_iter: int = None, C: float = None, random_state: int = None) -> LogisticRegression:
    """Create Logistic Regression classifier."""
    if max_iter is None:
        max_iter = config.LR_MAX_ITER
    if C is None:
        C = config.LR_C
    if random_state is None:
        random_state = config.RANDOM_STATE
    return LogisticRegression(max_iter=max_iter, C=C, random_state=random_state)


