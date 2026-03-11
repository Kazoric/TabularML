import torch

METRICS_REGISTRY = {}

def register_metric(name):
    def decorator(fn):
        METRICS_REGISTRY[name] = fn
        return fn
    return decorator

@register_metric("roc_auc")
def roc_auc_torch(y_true, y_pred):
    y_true = y_true.float().flatten()
    y_pred = y_pred.flatten()

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