from . import data_loader, preprocessing, models, evaluate
import joblib
import os
import logging
from . import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def train_pipeline(save_models=True):
    """
    Run the full training pipeline.
    
    Returns:
        dict: Dictionary containing trained models and scaler.
    """
    # Load Data
    logger.info("Loading data...")
    df = data_loader.load_data()
    
    # Split Data
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = preprocessing.split_data(df)
    
    # Scale Data
    logger.info("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(X_train, X_test)
    
    # Train KNN
    logger.info("Training KNN...")
    knn = models.create_knn_model()
    knn.fit(X_train_scaled, y_train)
    
    # Train SVM
    logger.info("Training SVM...")
    svm = models.create_svm_model()
    svm.fit(X_train_scaled, y_train)
    
    # Train Random Forest
    logger.info("Training Random Forest...")
    rf = models.create_rf_model()
    rf.fit(X_train_scaled, y_train)
    
    # Train Gradient Boosting
    logger.info("Training Gradient Boosting...")
    gb = models.create_gb_model()
    gb.fit(X_train_scaled, y_train)
    
    artifacts = {
        'knn': knn,
        'svm': svm,
        'rf': rf,
        'gb': gb,
        'scaler': scaler,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'y_train': y_train
    }
    
    if save_models:
        save_artifacts(artifacts)
        # Evaluate and save metrics
        for name, model in [('knn', knn), ('svm', svm), ('rf', rf), ('gb', gb)]:
             metrics = evaluate.evaluate_model(model, X_test_scaled, y_test, name.upper())
             evaluate.save_metrics_to_json(metrics, os.path.join(config.MODEL_DIR, f'{name}_metrics.json'))
        
    return artifacts

def save_artifacts(artifacts):
    """Save models and scaler to disk."""
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
        
    joblib.dump(artifacts['knn'], os.path.join(config.MODEL_DIR, 'knn_model.pkl'))
    joblib.dump(artifacts['svm'], os.path.join(config.MODEL_DIR, 'svm_model.pkl'))
    joblib.dump(artifacts['rf'], os.path.join(config.MODEL_DIR, 'rf_model.pkl'))
    joblib.dump(artifacts['gb'], os.path.join(config.MODEL_DIR, 'gb_model.pkl'))
    joblib.dump(artifacts['scaler'], os.path.join(config.MODEL_DIR, 'scaler.pkl'))
    logger.info(f"Artifacts saved to {config.MODEL_DIR}")

