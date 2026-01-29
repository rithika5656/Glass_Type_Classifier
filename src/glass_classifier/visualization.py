import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_matrix(df):
    """Plot correlation matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    return plt

def plot_class_distribution(y):
    """Plot distribution of target classes."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title("Class Distribution")
    plt.xlabel("Glass Type")
    plt.ylabel("Count")
    plt.tight_layout()
    return plt

def plot_confusion_matrix(cm, model_name="Model"):
    """Plot confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt
