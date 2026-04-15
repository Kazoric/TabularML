import torch

METRICS_REGISTRY = {}

def register_metric(name):
    def decorator(fn):
        METRICS_REGISTRY[name] = fn
        return fn
    return decorator

@register_metric("roc_auc")
def roc_auc_torch(y_true, y_pred):
    # 1. Handle y_pred dimensions (Logits -> Probs)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        # Apply Softmax and keep only the probability of the positive class (index 1)
        y_pred = torch.softmax(y_pred, dim=1)[:, 1]
    else:
        # Binary case with a single output neuron (Sigmoid)
        y_pred = torch.sigmoid(y_pred).flatten()

    y_true = y_true.float().flatten()

    sorted_indices = torch.argsort(y_pred, descending=True)
    y_true = y_true[sorted_indices]
    y_pred = y_pred[sorted_indices]

    tps = torch.cumsum(y_true, dim=0)
    fps = torch.cumsum(1 - y_true, dim=0)

    tps = torch.cat([torch.tensor([0.], device=y_true.device), tps])
    fps = torch.cat([torch.tensor([0.], device=y_true.device), fps])

    total_pos = tps[-1]
    total_neg = fps[-1]

    tpr = tps / total_pos
    fpr = fps / total_neg

    auc = torch.trapz(tpr, fpr)

    return auc

@register_metric("accuracy")
def accuracy_torch(y_true, y_pred):
    # Binary classification
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        y_pred_labels = (y_pred.flatten() >= 0.5).long()
    else:
        # Multi-classification
        y_pred_labels = torch.argmax(y_pred, dim=1)

    y_true_labels = y_true.flatten().long()
    correct = (y_pred_labels == y_true_labels).sum()
    accuracy = correct.float() / y_true_labels.numel()
    return accuracy

@register_metric("balanced_accuracy")
def balanced_accuracy_torch(y_true, y_pred):
    # 1. Get predicted labels (same logic as your accuracy function)
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        # For binary with sigmoid/logits
        y_pred_labels = (torch.sigmoid(y_pred).flatten() >= 0.5).long()
    else:
        # For multi-class (argmax)
        y_pred_labels = torch.argmax(y_pred, dim=1)

    y_true_labels = y_true.flatten().long()
    
    # 2. Identify unique classes
    classes = torch.unique(y_true_labels)
    recalls = []

    for cls in classes:
        # Mask for the current class
        true_is_cls = (y_true_labels == cls)
        pred_is_cls = (y_pred_labels == cls)

        # Count True Positives for this class
        tp = (true_is_cls & pred_is_cls).sum().float()
        # Count total actual samples for this class
        total_cls = true_is_cls.sum().float()

        # Recall for this class = TP / (TP + FN)
        # We add a tiny epsilon to avoid division by zero
        recalls.append(tp / (total_cls + 1e-8))

    # 3. Balanced Accuracy is the mean of recalls
    balanced_acc = torch.mean(torch.stack(recalls))
    return balanced_acc