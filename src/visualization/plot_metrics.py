import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from features.build_features import onehot_encoder


class ROCPlot:
    def __init__(self, y_test, y_pred_prob) -> None:
        self.y_test = y_test
        self.y_pred_prob = y_pred_prob
        self.y_test_enc, self.n_classes, self.categories = onehot_encoder(
            self.y_test)
        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()

    def roc_calc(self):
        for i in range(self.n_classes):
            self.fpr[i], self.tpr[i], _ = roc_curve(
                self.y_test_enc[:, i], self.y_pred_prob[:, i])
            self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
        return self.fpr, self.tpr, self.roc_auc

    def roc_plot(self):
        lw = 2

        fig = plt.figure(figsize=(10, 30))
        for i in range(len(self.categories)):
            ax = fig.add_subplot(6, 2, i+1)
            ax.plot(self.fpr[i], self.tpr[i], color='r',
                    lw=lw, label='ROC curve - (area = %0.2f)' % self.roc_auc[i])
            ax.plot([0, 1], [0, 1], color='b', lw=lw, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(
                "Receiver operating characteristic - {}".format(self.categories[i]))
            ax.legend(loc="lower right")
            ax.plot()
        return


class PrecisionRecallPlot:
    def __init__(self, y_test, y_pred_prob) -> None:
        self.y_test = y_test
        self.y_pred_prob = y_pred_prob
        self.y_test_enc, self.n_classes, self.categories = onehot_encoder(
            self.y_test)
        self.precision = dict()
        self.recall = dict()
        self.average_precision = dict()

    def precision_recall_calc(self):
        for i in range(self.n_classes):
            self.precision[i], self.recall[i], _ = precision_recall_curve(
                self.y_test_enc[:, i], self.y_pred_prob[:, i])
            self.average_precision[i] = average_precision_score(
                self.y_test_enc[:, i], self.y_pred_prob[:, i])

        # A "micro-average": quantifying score on all classes jointly
        self.precision["micro"], self.recall["micro"], _ = precision_recall_curve(
            self.y_test_enc.ravel(), self.y_pred_prob.ravel()
        )
        self.average_precision["micro"] = average_precision_score(
            self.y_test_enc, self.y_pred_prob, average="micro")

        display = PrecisionRecallDisplay(
            recall=self.recall["micro"],
            precision=self.precision["micro"],
            average_precision=self.average_precision["micro"],
        )
        display.plot()
        _ = display.ax_.set_title("Micro-averaged over all classes")

        return self.precision, self.recall, self.average_precision

    def precision_recall_plot(self):
        colors = cycle(["navy", "turquoise", "darkorange",
                        "cornflowerblue", "teal"])
        _, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=self.recall["micro"],
            precision=self.precision["micro"],
            average_precision=self.average_precision["micro"],
        )

        display.plot(
            ax=ax, name="Micro-average precision-recall", color="gold")

        for i, color in zip(range(self.n_classes), colors):
            display = PrecisionRecallDisplay(
                recall=self.recall[i],
                precision=self.precision[i],
                average_precision=self.average_precision[i],
            )
            display.plot(
                ax=ax, name=f"Precision-recall for class {self.categories[i]}", color=color)
        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Extension of Precision-Recall curve to multi-class")

        plt.show()
        return
