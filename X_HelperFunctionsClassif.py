# Functions for TS classification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion(y_true, y_pred, labels, title_str):
  """
  Plots a confusion matrix heatplot given the multiclass predictions and actual
  classes.
  """
  matrix = confusion_matrix(y_true, y_pred, labels = labels)
  _ = sns.heatmap(
    matrix, xticklabels = labels, yticklabels = labels, cmap = "Reds", 
    annot = True, fmt = "g", square = True, cbar = False, linecolor = "black", 
    linewidths = 0.5)
  _ = plt.xlabel("Predicted classes")
  _ = plt.ylabel("True classes")
  _ = plt.title("Confusion matrix, " + title_str)
  plt.show()
  plt.close("all")
