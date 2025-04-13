from sklearn.metrics import confusion_matrix

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0][0] if cm.shape[0] > 1 else 0
    fp = cm[0][1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity