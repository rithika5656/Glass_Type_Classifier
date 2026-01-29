from . import data_loader, preprocessing, models
import joblib
import os
from . import config

def train_pipeline(save_models=True):
    """
    Run the full training pipeline.
    
    Returns:
        dict: Dictionary containing trained models and scaler.
    """
    # Load Data
    print("Loading data...")
    df = data_loader.load_data()
    
    # Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = preprocessing.split_data(df)
    
    # Scale Data
    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(X_train, X_test)
    
    # Train KNN
    print("Training KNN...")
    knn = models.create_knn_model()
    knn.fit(X_train_scaled, y_train)
    
    # Train SVM
    print("Training SVM...")
    svm = models.create_svm_model()
    svm.fit(X_train_scaled, y_train)
    
    artifacts = {
        'knn': knn,
        'svm': svm,
        'scaler': scaler,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'y_train': y_train
    }
    
    if save_models:
        save_artifacts(artifacts)
        
    return artifacts

def save_artifacts(artifacts):
    """Save models and scaler to disk."""
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
        
    joblib.dump(artifacts['knn'], os.path.join(config.MODEL_DIR, 'knn_model.pkl'))
    joblib.dump(artifacts['svm'], os.path.join(config.MODEL_DIR, 'svm_model.pkl'))
    joblib.dump(artifacts['scaler'], os.path.join(config.MODEL_DIR, 'scaler.pkl'))
    print(f"Artifacts saved to {config.MODEL_DIR}")
