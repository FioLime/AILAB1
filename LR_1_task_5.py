import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

confusion_matrix(df.actual_label.values, df.predicted_RF.values)


def valko_find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))


def valko_find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))


def valko_find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))


def valko_find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))


print('TP: ', valko_find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN: ', valko_find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP: ', valko_find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN: ', valko_find_TN(df.actual_label.values, df.predicted_RF.values))


def valko_find_conf_matrix_values(y_true, y_pred):
    TP = valko_find_TP(y_true, y_pred)
    FN = valko_find_FN(y_true, y_pred)
    FP = valko_find_FP(y_true, y_pred)
    TN = valko_find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def valko_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = valko_find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


valko_confusion_matrix(df.actual_label.values, df.predicted_RF.values)
assert np.array_equal(valko_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values,
                                       df.predicted_RF.values)), 'valko_confusion_matrix() is not correct for RF'
assert np.array_equal(valko_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values,
                                       df.predicted_LR.values)), 'valko_confusion_matrix() is not correct for LR'

accuracy_score(df.actual_label.values, df.predicted_RF.values)


def valko_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = valko_find_conf_matrix_values(y_true, y_pred)
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    return accuracy


assert valko_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values,
                                                                                              df.predicted_RF.values), 'valko_accuracy_score failed on RF'

assert valko_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values,
                                                                                              df.predicted_LR.values), 'valko_accuracy_score failed on LR'

print('Accuracy RF: %.3f' % (valko_accuracy_score(df.actual_label.values, df.predicted_RF.values)))

recall_score(df.actual_label.values, df.predicted_RF.values)


def valko_recall_score(y_true, y_pred):
    TP, FN, FP, TN = valko_find_conf_matrix_values(y_true, y_pred)
    recall = TP / (TP + FN)
    return recall


assert valko_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values,
                                                                                          df.predicted_RF.values), 'valko_recall_score failed on RF'

assert valko_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values,
                                                                                          df.predicted_LR.values), 'valko_recall_score failed on LR'

print('Recall RF: %.3f' % (valko_recall_score(df.actual_label.values, df.predicted_RF.values)))

precision_score(df.actual_label.values, df.predicted_RF.values)


def valko_precision_score(y_true, y_pred):
    TP, FN, FP, TN = valko_find_conf_matrix_values(y_true, y_pred)
    precision = TP / (TP + FP)
    return precision


assert valko_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values,
                                                                                                df.predicted_RF.values), 'valko_precision_score failed on RF'

assert valko_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values,
                                                                                                df.predicted_LR.values), 'valko_precision_score failed on LR'

print('Precision RF: %.3f' % (valko_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f' % (valko_precision_score(df.actual_label.values, df.predicted_LR.values)))

f1_score(df.actual_label.values, df.predicted_RF.values)


def valko_f1_score(y_true, y_pred):
    recall = valko_recall_score(y_true, y_pred)
    precision = valko_precision_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# Перевірка для RF
assert valko_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values,
                                                                                  df.predicted_RF.values), 'valko_f1_score failed on RF'

# Перевірка для LR
assert valko_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values,
                                                                                  df.predicted_LR.values), 'valko_f1_score failed on LR'

print('F1 RF: %.3f' % (valko_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (valko_f1_score(df.actual_label.values, df.predicted_LR.values)))

print('scores with threshold = 0.5')
print('Accuracy RF: %.3f' % (valko_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (valko_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (valko_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (valko_f1_score(df.actual_label.values, df.predicted_RF.values)))

print('')
print('Scores with threshold = 0.25')
print('Accuracy RF: %.3f' % (valko_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (valko_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print(
    'Precision RF: %.3f' % (valko_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (valko_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF, 'r- ', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

print('AUC RF:%.3f' % auc_RF)
print('AUC LR:%.3f' % auc_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
