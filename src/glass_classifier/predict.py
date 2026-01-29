import joblib
import os
import numpy as np
from . import config

def load_models():
    """Load trained models and scaler."""
    try:
        knn = joblib.load(os.path.join(config.MODEL_DIR, 'knn_model.pkl'))
        svm = joblib.load(os.path.join(config.MODEL_DIR, 'svm_model.pkl'))
        rf = joblib.load(os.path.join(config.MODEL_DIR, 'rf_model.pkl'))
        gb = joblib.load(os.path.join(config.MODEL_DIR, 'gb_model.pkl'))
        lr = joblib.load(os.path.join(config.MODEL_DIR, 'lr_model.pkl'))
        scaler = joblib.load(os.path.join(config.MODEL_DIR, 'scaler.pkl'))
        return knn, svm, rf, gb, lr, scaler
    except FileNotFoundError:
        print("Models not found. Please train first.")
        return None, None, None, None, None, None



def predict_sample(models_dict, sample_data):
    """
    Predict class for a single sample.
    
    Args:
        models_dict (dict): Dictionary with 'knn', 'svm', 'rf', 'gb', 'scaler'.
        sample_data (list or np.array): Feature values.
        
    Returns:
        dict: Predictions from all models.
    """
    scaler = models_dict['scaler']
    sample_scaled = scaler.transform(np.array(sample_data).reshape(1, -1))
    
    knn_pred = models_dict['knn'].predict(sample_scaled)[0]
    svm_pred = models_dict['svm'].predict(sample_scaled)[0]
    rf_pred = models_dict['rf'].predict(sample_scaled)[0]
    gb_pred = models_dict['gb'].predict(sample_scaled)[0]
    lr_pred = models_dict['lr'].predict(sample_scaled)[0]
    
    return {
        'KNN': int(knn_pred),
        'SVM': int(svm_pred),
        'Random Forest': int(rf_pred),
        'Gradient Boosting': int(gb_pred),
        'Logistic Regression': int(lr_pred)
    }


