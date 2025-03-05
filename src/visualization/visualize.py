import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np

def plot_roc_auc(y_test, y_pred_probs, classes):
    """
    Plots the AUC-ROC curve for a multi-class classification model.
    
    Args:
        y_test (array): True labels.
        y_pred_probs (array): Predicted probabilities for each class.
        classes (list): List of class labels.
    """
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    
    plt.figure(figsize=(8, 6))
    
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {auc_score:.4f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_labels):
    """
    Plots a confusion matrix.

    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        class_labels (list): List of class labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

def compute_f1_score(y_true, y_pred):
    """
    Computes and prints the F1-score.

    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
    """
    f1 = f1_score(y_true, y_pred, average="macro")  # Macro-average F1-score
    print(f"Final F1-Score: {f1:.4f}")
    return f1
