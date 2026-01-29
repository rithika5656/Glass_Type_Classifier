from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import json
import os


def evaluate_model(model, X_test_scaled, y_test, model_name="Model"):
    """
    Evaluate a model and return metrics.
    
    Args:
        model: Trained model.
        X_test_scaled: Scaled test features.
        y_test: True labels.
        model_name: Name of model for printing.
        
    Returns:
        dict: Metrics dictionary.
    """
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    # print(classification_report(y_test, y_pred)) # Optional verbose output
    
    return {
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm
    }

def save_metrics_to_json(metrics, filepath):
    """Save metrics to a JSON file."""
    # Convert numpy types to native types for JSON serialization
    serialized = {}
    for k, v in metrics.items():
        if k == 'confusion_matrix':
            serialized[k] = v.tolist()
        else:
            serialized[k] = v
            
    with open(filepath, 'w') as f:
        json.dump(serialized, f, indent=4)

def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation."""
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"CV Scores: {scores}")
    print(f"Mean CV Score: {scores.mean():.4f}")
    return scores.mean()

