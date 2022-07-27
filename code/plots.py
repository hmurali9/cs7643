import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):

  fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

  ax1.plot(train_losses, label="Training Loss")
  ax1.plot(valid_losses, label="Validation Loss")
  ax1.set_ylabel("Loss")
  ax1.set_xlabel("epoch")
  ax1.set_title("Loss Curve")
  ax1.legend()

  ax2.plot(train_accuracies, label="Training Accuracy")
  ax2.plot(valid_accuracies, label="Validation Accuracy")
  ax2.set_ylabel("Accuracy")
  ax2.set_xlabel("epoch")
  ax2.set_title("Accuracy Curve")
  ax2.legend()

  plt.show()
  fig.savefig('../output/MLP_Loss_Accuracy_Plots.svg', format='svg')

  pass


def plot_confusion_matrix(results, class_names):

  y_true = np.array(list(list(zip(*results))[0])).astype(int)
  y_pred = np.array(list(list(zip(*results))[1])).astype(int)

  d = [0, 1, 2, 3, 4]

  cm = confusion_matrix(y_true, y_pred, labels=d)

  #Calculate precision, recall and F1-score

  prec = []
  rec = []
  acc = []

  for i in range(5):
      prec.append(cm[i,i]/np.sum(cm[:,i]))
      rec.append(cm[i,i]/np.sum(cm[i,:]))
      acc.append((cm[i,i] + np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1)))/np.sum(cm))

  prec = np.array(prec)
  rec = np.array(rec)
  f1 = (2*prec*rec)/(prec+rec)

  #Create confusion table
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp.plot()
  plt.show()
  disp.figure_.savefig('../output/MLP_Confusion_Matrix.svg', format='svg')

  #Display results table
  df = pd.DataFrame({'Labels': class_names, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1, 'Count': np.bincount(y_true)})
  display(df.round(3))

  pass
