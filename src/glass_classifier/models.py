from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from . import config

def create_knn_model(n_neighbors=None):
    """Create KNN classifier."""
    if n_neighbors is None:
        n_neighbors = config.KNN_N_NEIGHBORS
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def create_svm_model(kernel=None, random_state=None):
    """Create SVM classifier."""
    if kernel is None:
        kernel = config.SVM_KERNEL
    if random_state is None:
        random_state = config.RANDOM_STATE
    return SVC(kernel=kernel, random_state=random_state)
