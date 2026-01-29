from src.glass_classifier import models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression



def test_create_knn():
    model = models.create_knn_model(n_neighbors=5)
    assert isinstance(model, KNeighborsClassifier)
    assert model.n_neighbors == 5

def test_create_svm():
    model = models.create_svm_model(kernel='linear')
    assert isinstance(model, SVC)
    assert model.kernel == 'linear'

def test_create_rf():
    model = models.create_rf_model(n_estimators=10)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10

def test_create_gb():
    model = models.create_gb_model(n_estimators=10)
    assert isinstance(model, GradientBoostingClassifier)
    assert model.n_estimators == 10

def test_create_lr():
    model = models.create_lr_model(max_iter=500)
    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 500


