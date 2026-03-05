import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from utils.viz import plot_confusion_matrix, plot_roc_curve

def evaluate_global_model(model, dataloader, device, output_dir="EXPERIMENT"):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)[:, 1] # prob of class '1'
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    cm = confusion_matrix(all_targets, all_preds)
    
    # Compute ROC only if there's enough classes represented
    if len(set(all_targets)) > 1:
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = [0], [0], 0.0
    
    os.makedirs(output_dir, exist_ok=True)
    plot_confusion_matrix(cm, classes=['Benign', 'Malicious'], save_path=os.path.join(output_dir, "confusion_matrix.png"))
    plot_roc_curve(fpr, tpr, roc_auc, save_path=os.path.join(output_dir, "roc_curve.png"))
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
