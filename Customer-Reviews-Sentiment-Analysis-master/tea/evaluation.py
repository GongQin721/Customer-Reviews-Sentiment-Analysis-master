from itertools import cycle

import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy import interp
from sklearn import metrics
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, \
    precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize

plt.rcParams['figure.figsize'] = (16, 8)


def create_confusion_matrix(actual, predicted, category):
    """
    Calculates the confusion matrix for a give category.
    :param actual: The actual labels of the data
    :param predicted: The predicted labels of the data
    :param category: The category we of the confusion matrix
    :return: dictionary, with the values of the confusion matrix
    """
    conf_matrix = dict()
    conf_matrix['TP'], conf_matrix['FP'], conf_matrix['TN'], conf_matrix['FN'] = 0, 0, 0, 0

    print('The category is: {}'.format(category))
    for sentence in predicted:
        if sentence in actual[predicted[sentence]] and predicted[sentence] == category:
            print('TP: Actual: {}, Predicted: {}'.format(category, category))
            conf_matrix['TP'] += 1
        elif sentence in actual[predicted[sentence]]:
            print('TN: Actual: not category, Predicted: not category'.format(predicted[sentence]))
            conf_matrix['TN'] += 1
        elif sentence not in actual[predicted[sentence]] and predicted[sentence] == category:
            print('FP: Actual: not category, Predicted: {}'.format(category))
            conf_matrix['FP'] += 1
        else:
            print('FN: Actual: {}, Predicted: {}'.format(category, predicted[sentence]))
            conf_matrix['FN'] += 1

    return conf_matrix


def calculate_evaluation_metrics(confusion_matrix):
    """
    Calculates the evaluation metrics of the model.
    :param confusion_matrix: The confusion matrix of the model.
    :return: dictionary, with the metrics of the model.
    """
    metrics = dict()

    metrics['precision'] = confusion_matrix.get('TP', 1) / (
            confusion_matrix.get('TP', 1) + confusion_matrix.get('FP', 1))
    metrics['recall'] = confusion_matrix.get('TP', 1) / (
            confusion_matrix.get('TP', 1) + confusion_matrix.get('FN', 1))
    metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])

    return metrics


def create_clf_report(y_true, y_pred, classes):
    """
    This function calculates several metrics about a classifier and creates a mini report.

    :param y_true: iterable. An iterable of string or ints.
    :param y_pred: iterable. An iterable of string or ints.
    :param classes: iterable. An iterable of string or ints.
    :return: dataframe. A pandas dataframe with the confusion matrix.
    """
    confusion = pd.DataFrame(confusion_matrix(y_true, y_pred),
                             index=classes,
                             columns=['predicted_{}'.format(c) for c in classes])

    print("-" * 80, end='\n')
    print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    print("-" * 80)

    print("Confusion Matrix:", end='\n\n')
    print(confusion)

    print("-" * 80, end='\n')
    print("Classification Report:", end='\n\n')
    print(classification_report(y_true, y_pred, digits=3), end='\n')

    return confusion


def benchmark(clf, train_X, train_y, test_X, test_y):
    """
    This function calculates metrics for evaluation of a classifier over a training and a test set.

    :param clf: obj. An sklearn classifier
    :param train_X: array
    :param train_y: array
    :param test_X: array
    :param test_y: array
    :return: dict
    """
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)

    f1 = metrics.f1_score(test_y, pred, average='weighted')

    accuracy = metrics.accuracy_score(test_y, pred)

    print(" Acc: %f " % accuracy)

    result = {'f1': f1, 'accuracy': accuracy, 'train size': len(train_y),
              'test size': len(test_y), 'predictions': pred}

    return result


def create_benchmark_plot(train_X,
                          train_y,
                          test_X,
                          test_y,
                          clf,
                          splits=20,
                          plot_outfile=None,
                          y_ticks=0.025,
                          min_y_lim=0.4):
    """
    This method creates a benchmark plot.

    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :param clf:
    :param splits:
    :param plot_outfile:
    :param y_ticks:
    :param min_y_lim:
    :return:
    """

    results = {'train_size': [],
               'on_test': [],
               'on_train': []}

    # splitting the train X in n (almost) equal splits)
    train_x_splits = np.array_split(ary=train_X, indices_or_sections=splits, axis=0)

    # splitting the train y in the same splits as the train X
    train_y_splits = np.array_split(ary=train_y, indices_or_sections=splits, axis=0)

    # setting parameters for the graph.
    font_p = FontProperties()

    font_p.set_size('small')

    fig = plt.figure()
    fig.suptitle('Learning Curves', fontsize=20)

    ax = fig.add_subplot(111)
    ax.axis(xmin=0, xmax=train_X.shape[0] * 1.05, ymin=0, ymax=1.1)

    plt.xlabel('N. of training instances', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)

    plt.grid(True)

    plt.axvline(x=int(train_X.shape[0] * 0.3))
    plt.yticks(np.arange(0, 1.025, 0.025))

    if y_ticks == 0.05:
        plt.yticks(np.arange(0, 1.025, 0.05))

    elif y_ticks == 0.025:
        plt.yticks(np.arange(0, 1.025, 0.025))

    plt.ylim([min_y_lim, 1.025])

    # each time adds up one split and refits the model.
    for i in range(1, splits + 1):
        train_x_part = np.concatenate(train_x_splits[:i])
        train_y_part = np.concatenate(train_y_splits[:i])

        print(20 * '*')
        print('Split {} size: {}'.format(i, train_x_part.shape))

        results['train_size'].append(train_x_part.shape[0])

        result_on_test = benchmark(clf=clf,
                                   train_X=train_x_part,
                                   train_y=train_y_part,
                                   test_X=test_X,
                                   test_y=test_y)

        # calculates each time the metrics also on the test.
        results['on_test'].append(result_on_test['accuracy'])

        # calculates the metrics for the given training part
        result_on_train_part = benchmark(clf=clf,
                                         train_X=train_x_part,
                                         train_y=train_y_part,
                                         test_X=train_x_part,
                                         test_y=train_y_part)

        results['on_train'].append(result_on_train_part['accuracy'])

        line_up, = ax.plot(results['train_size'],
                           results['on_train'],
                           'o-',
                           label='Accuracy on Train')

        line_down, = ax.plot(results['train_size'],
                             results['on_test'],
                             'o-',
                             label='Accuracy on Test')

        plt.legend([line_up, line_down],
                   ['Accuracy on Train', 'Accuracy on Test'],
                   prop=font_p)

    if plot_outfile:
        fig.savefig(plot_outfile)

    plt.show()

    return results


def prec_recall_multi(n_classes, X_test, Y_test, fitted_clf):
    """

    :param n_classes: int. the number of classes of the data
    :param X_test: list
    :param Y_test: list
    :param fited_clf: fitted classifier
    :return: 3 dictionaries for precision recall, average_precision
    """
    y_score = fitted_clf.decision_function(X_test)
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())

    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
    return precision, recall, average_precision


def plot_micro_prec_recall(precision, recall, average_precision):
    """
    Created an average precision plot, micro-averaged over all classes
    :param precision: dict.
    :param recall: dict.
    :param average_precision: dict.
    :return: plot
    """
    sns.set()
    sns.set_style("dark")
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')

    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(
        average_precision["micro"]),
        fontsize=24)

    plt.show()


def plot_micro_prec_recall_per_class(n_classes, precision, recall, average_precision):
    """

    :param n_classes: number of classes
    :param precision: dict.
    :param recall: dict.
    :param average_precision: dict.
    :return: plot
    """

    sns.set()
    sns.set_style("dark")

    pal = sns.color_palette("cubehelix", 5)
    colors = cycle(pal.as_hex())

    plt.figure(figsize=(14, 10))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title('Extension of Precision-Recall curve to multi-class', fontsize=24)
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.show()


def compute_roc_curve_area(n_classes, X_test, y_test, fittedclf):
    """
    Computes the roc curve and roc area for a multiclass problem
    :param n_classes: number of classes
    :param X_test: list
    :param y_test: list
    :param fittedclf: fitted classifier
    :return: 3 dictionaries
    """
    y_score = fittedclf.decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc


def plot_roc_single(fpr, tpr, roc_auc, nclass):
    """
    Plot roc for a single class
    :param fpr: dict
    :param tpr: dict
    :param roc_auc: dict
    :param nclass: int number of class
    :return:
    """
    plt.figure()
    lw = 2
    plt.plot(fpr[nclass], tpr[nclass], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[nclass])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_multi(fpr, tpr, roc_auc, n_classes):
    """
    Plots roc curves for multiclass
    :param fpr: dict
    :param tpr: dict
    :param roc_auc: dict
    :param n_classes: int number of classes
    :return:
    """

    lw = 2
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves

    sns.set()
    sns.set_style("dark")

    pal = sns.color_palette("cubehelix", 3)
    colors = cycle(pal.as_hex())

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_binary(X, y, fitted_clf):
    """

    :param X:
    :param y:
    :param fitted_clf:
    :return:
    """
    logit_roc_auc = roc_auc_score(y, fitted_clf.predict(X))

    fpr, tpr, thresholds = roc_curve(y, fitted_clf.predict_proba(X)[:, 1])

    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    plt.savefig('Log_ROC')
    plt.show()


if __name__ == "__main__":
    a = ['positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive',
         'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative']

    b = ['negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'positive',
         'positive', 'negative', 'negative', 'negative', 'negative', 'negative', 'positive']

    create_clf_report(y_true=a, y_pred=b, classes=['positive', 'negative'])

    # Another Example

    digits = datasets.load_digits()
    X = digits.data
    Y = digits.target

    # split 80% train , 20% test
    digits_train_x = X[0: int(X.shape[0] * 0.8), :]
    digits_train_y = Y[0: int(X.shape[0] * 0.8)]
    digits_test_x = X[int(X.shape[0] * 0.8):, :]
    digits_test_y = Y[int(X.shape[0] * 0.8):]

    # normalize values between -1 and 1 with the simple minmax algorithm
    normalizer = MinMaxScaler(feature_range=(-1.0, 1.0))
    digits_train_x = normalizer.fit_transform(digits_train_x)
    digits_test_x = normalizer.transform(digits_test_x)

    from sklearn.utils import shuffle

    train_x_shuffled, train_y_shuffled = shuffle(digits_train_x,
                                                 digits_train_y,
                                                 random_state=1989)

    clf_svm = svm.LinearSVC(random_state=1989,
                            C=100.,
                            penalty='l2',
                            max_iter=1000)

    create_benchmark_plot(train_x_shuffled,
                          train_y_shuffled,
                          digits_test_x,
                          digits_test_y,
                          clf_svm,
                          10,
                          None,
                          0.025,
                          0.5)

    # Another example for plotting Precision-recall curves for multiclass

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Add noisy features
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    # Use label_binarize to be multi-label like settings
    Y = label_binarize(y, classes=[0, 1, 2])
    n_classes = Y.shape[1]
    # Split into training and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5, random_state=random_state)

    # We use OneVsRestClassifier for multi-label prediction
    from sklearn.multiclass import OneVsRestClassifier

    # Run classifier
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
    classifier.fit(X_train, Y_train)

    # Learn to predict each class against the other
    classifier2 = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    classifier2.fit(X_train, Y_train)

    # Now plotting begins

    prec, recall, av_prec = prec_recall_multi(n_classes, X_test, Y_test, classifier)
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(av_prec["micro"]))

    plot_micro_prec_recall(prec, recall, av_prec)

    plot_micro_prec_recall_per_class(n_classes, prec, recall, av_prec)

    fprdict, tprdict, roc_aucdict = compute_roc_curve_area(n_classes, X_test, Y_test, classifier2)

    plot_roc_single(fprdict, tprdict, roc_aucdict, 2)

    plot_roc_multi(fprdict, tprdict, roc_aucdict, n_classes)
