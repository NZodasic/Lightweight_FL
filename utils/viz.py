import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_training_curves(history, save_path):
    rounds = [x['round'] for x in history]
    accs = [x['val_accuracy'] for x in history]
    
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accs, marker='o', label='Val Accuracy')
    plt.title('Federated Training Accuracy')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
