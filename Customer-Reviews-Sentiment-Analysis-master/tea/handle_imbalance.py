import random

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample, shuffle

from tea import setup_logger

logger = setup_logger(__name__)


def undersample(train_X, train_y):
    """
    If the class with the minimum number of instances has n instances then all classes should
    have n instances where we randomly select n instances from the classes with more instances than n.

    :param train_X:
    :param train_y:
    :return:
    """
    # the unique labels in classes
    unique = set(train_y)

    # find the class with the minimum number of instances
    minimum = train_y.size

    for i in unique:
        if train_y[train_y == i].size < minimum:
            minimum = train_y[train_y == i].size

    ret_x = np.empty(shape=[0, train_X[0].size])
    ret_y = np.empty(shape=[0, 1])

    # Create a new matrix with "minimum" number of instances for each category
    for i in unique:

        temp = np.asarray(random.sample(train_X[train_y == i].tolist(), minimum))
        ret_x = np.append(ret_x, temp, axis=0)

        for j in range(0, minimum):
            ret_y = np.append(ret_y, i)

    return np.asarray(ret_x), ret_y


def oversample(train_X, train_y):
    """

    If the class with the Maximum number of instances has n instances then all classes should
    have n instances where we randomly select n instances from the classes with less instances than n

    :param train_X:
    :param train_y:
    :return:
    """

    unique = set(train_y)
    # find maximum num of instances

    maximum = 0
    for i in unique:
        if train_y[train_y == i].size > maximum:
            maximum = train_y[train_y == i].size

    ret_x = np.empty(shape=[0, train_X[0].size])
    ret_y = np.empty(shape=[0, 1])

    for i in unique:
        to_go = maximum

        # if one class has k instances and n div k > 1
        # then add n div k times all the k instances

        while to_go >= train_X[train_y == i][:, 0].size:
            ret_x = np.append(ret_x, train_X[train_y == i], axis=0)
            to_go -= train_X[train_y == i][:, 0].size
        # then add n mod k instances to reach maximum number

        if to_go > 0:
            temp = np.asarray(random.sample(train_X[train_y == i].tolist(), to_go))
            ret_x = np.append(ret_x, temp, axis=0)

        for j in range(0, maximum):
            ret_y = np.append(ret_y, i)

    return np.asarray(ret_x), ret_y


class SimpleMinorityClassOversampler(BaseEstimator, TransformerMixin):

    def __init__(self,
                 column,
                 minority_class,
                 n_samples=5,
                 random_state=123,
                 reshuffle=True):
        """
        1. Over sample Minority Class. Over-sampling is the process of randomly duplicating observations
        from the minority class in order to reinforce its signal. The most common way is to simply re-sample
        with replacement.

        2. We create a new DataFrame with an over sampled minority class. Here are the steps:
          2.1 We'll separate observations from each class into different DataFrames.
          2.2 We'll re-sample the minority class with replacement, setting the number of samples
              to match that of the majority class.
          2.3 Finally, we'll combine the up-sampled minority class DataFrame with the original majority class DataFrame.

        :param column:
        :param minority_class:
        :param n_samples:
        :param random_state:
        """

        self.column = column
        self.n_samples = n_samples
        self.minority_class = minority_class
        self.random_state = random_state
        self.reshuffle = reshuffle

    def transform(self, X, y=None):
        if y is None:
            raise Exception('Must provide labels in order to resample')

        df = pd.concat([X, y], axis=1)

        df_minority = df[df[self.column] == self.minority_class].copy()

        # Over-sampling minority class
        minority_samples_df = resample(df_minority,
                                       replace=True,  # sample with replacement
                                       n_samples=self.n_samples,
                                       random_state=self.random_state)  # reproducible results

        # Combine majority class with upsampled minority class
        df_oversampled = pd.concat([df, minority_samples_df])
        if self.reshuffle:
            df_oversampled = shuffle(df_oversampled, random_state=self.random_state)


        logger.info(df_oversampled[self.column].value_counts() / len(df_oversampled))

        return df_oversampled.drop([self.column], axis=1), df_oversampled[self.column]

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""

        return self


if __name__ == "__main__":
    t_list = ['this', 'is', 'a', 'very', 'big',
              'sentence', 'for', 'illustration', 'purposes', 'only',
              'this', 'is', 'a', 'very', 'big',
              'sentence', 'for', 'illustration', 'purposes', 'only']

    p_list = ['positive', 'negative', 'neutral', 'neutral', 'positive',
              'positive', 'negative', 'negative', 'positive', 'negative',
              'positive', 'negative', 'neutral', 'neutral', 'positive',
              'positive', 'negative', 'negative', 'positive', 'negative']

    X = pd.DataFrame({'text': t_list})
    y = pd.DataFrame({'polarity': p_list})

    obj = SimpleMinorityClassOversampler(column='polarity',
                                         minority_class='neutral',
                                         n_samples=5,
                                         reshuffle=True)

    print(obj.transform(X, y))
