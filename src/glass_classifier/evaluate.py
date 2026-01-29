from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
