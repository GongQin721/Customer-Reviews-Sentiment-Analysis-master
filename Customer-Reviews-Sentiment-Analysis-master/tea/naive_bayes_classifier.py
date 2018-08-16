import string
import re
import math
import operator


class NaiveBayesClassifier(object):
    """Implementation of Naive Bayes for binary classification"""

    def __init__(self):
        self.classes = set()
        self.log_class_priors = {}
        self.word_counts = {}
        self.voc = set()

    @staticmethod
    def clean(text):
        """
        Cleans up a sequence/string by removing punctuation.
        :param text: string, the give sequence.
        :return: string, a cleaned sequence.
        """
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def tokenize(self, text):
        """
        Tokenizes a sequence/string into words.
        :param text: string, the give sequence.
        :return: a list with the tokens of the sequence.
        """
        text = self.clean(text).lower()
        return re.split('\W+', text)

    @staticmethod
    def get_word_counts(words):
        """
        Counts up how many of each word appears in a list of words.
        :param words: a list of strings.
        :return: a dictionary with the words and their frequency.
        """
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    def fit(self, X, Y):
        """
        Finds the classes of the dataset and computes log class priors. For each (document, label) pair, tokenize
        the document into words, add it to the vocabulary for each class and update the number of counts. Also add
        that word to the global vocabulary.
        :param X: a list with sentences
        :param Y: a list with the corresponding classes
        """
        self.classes = set(Y)

        n = len(X)
        for target_class in self.classes:
            self.log_class_priors[str(target_class)] = math.log(sum(1 for label in Y if label == target_class) / n)
            self.word_counts[str(target_class)] = {}

        for x, y in zip(X, Y):
            for target_class in self.classes:
                if y == target_class:
                    c = str(target_class)

            counts = self.get_word_counts(self.tokenize(x))
            for word, count in counts.items():
                if word not in self.voc:
                    self.voc.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0

                self.word_counts[c][word] += count

    def predict(self, X):
        """
        Applies Naive Bayes directly. For each document, it iterates each of the words, computes the log likelihood,
        and sum them all up for each class. Then it adds the log class priors so as to compute the posterior and checks
        to see which score is bigger for that document.
        :param X: a list with sentences.
        :return: a list with the predicted labels.
        """
        result = []
        for x in X:
            counts = self.get_word_counts(self.tokenize(x))
            posterior = dict()

            # initialize posterior dictionary
            for target_class in self.classes:
                posterior[str(target_class)] = 0

            for word, _ in counts.items():
                if word not in self.voc:
                    continue

                # add Laplace smoothing
                log_w_given_class = dict()
                for target_class in self.classes:
                    log_w_given_class[str(target_class)] = math.log(
                        (self.word_counts[str(target_class)].get(word, 0.0) + 1) / (
                            sum(self.word_counts[str(target_class)].values()) + len(self.voc)))

                    posterior[str(target_class)] += log_w_given_class[str(target_class)]

            # add log priors to compute posterior for each class
            for target_class in self.classes:
                posterior[str(target_class)] += self.log_class_priors[str(target_class)]

            result.append(int(max(posterior.items(), key=operator.itemgetter(1))[0]))

        return result

if __name__ == '__main__':
    train_data = ['the most fun film of the summer',
                  'very powerful',
                  'just plain boring',
                  'entirely predictable and lacks energy',
                  'no surprises and very few laughs',
                  'average performance',
                  'average performance']

    train_target = [1, 1, 0, 0, 0, 2, 2]

    model = NaiveBayesClassifier()
    model.fit(train_data, train_target)

    test_data = ['predictable with no fun',
                 'predictable with few fun',
                 'very very fun',
                 'very fun',
                 'average']

    test_target = [0, 0, 1, 0, 2]

    pred = model.predict(test_data)

    accuracy = sum(1 for i in range(len(pred)) if pred[i] == test_target[i]) / float(len(pred))
    print("{0:.4f}".format(accuracy))
