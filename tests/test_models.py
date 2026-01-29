from src.glass_classifier import models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def test_create_knn():
    model = models.create_knn_model(n_neighbors=5)
    assert isinstance(model, KNeighborsClassifier)
    assert model.n_neighbors == 5

def test_create_svm():
    model = models.create_svm_model(kernel='linear')
    assert isinstance(model, SVC)
    assert model.kernel == 'linear'
