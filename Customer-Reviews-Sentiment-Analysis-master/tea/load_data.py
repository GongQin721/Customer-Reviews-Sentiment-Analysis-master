import re
from collections import Counter

import pandas as pd
from bs4 import BeautifulSoup as Soup
from bs4 import Tag
from sklearn.model_selection import StratifiedShuffleSplit

from tea import DATA_DIR, setup_logger

logger = setup_logger(__name__)

TRAIN_FILE = 'ABSA16_Laptops_Train_SB1_v2.xml'
TEST_FILE = 'EN_LAPT_SB1_TEST_.xml.gold'


def calculate_label_ratio(labels):
    """

    :param labels:
    :return:
    """

    x = Counter(labels)
    sum_counts = sum(x.values())

    print()
    for t in x.most_common():
        ratio = round(t[1] / sum_counts * 100, 2)

        print('Label: {}, Instances: {}, Ratio: {}%'.format(t[0], t[1], ratio))


def parse_reviews(file_type='train',
                  save_data=True,
                  load_data=True):
    """

    :param file_type: str. the file type. Enum between 'train' and 'test'
    :param save_data: bool. whether the extracted data will be saved
    :param load_data: bool. whether the extracted data should be loaded
    :return: pandas data-frame with 2 columns: polarity, text
    """
    assert file_type in ['train', 'test']

    file = TRAIN_FILE if file_type == 'train' else TEST_FILE

    path = "{}{}".format(DATA_DIR, file)

    if load_data:
        try:
            x = path.split('.')[-1]
            infile = re.sub(x, 'csv', path)
            logger.info('Loading file: {}'.format(infile))

            return pd.read_csv(infile)
        except FileNotFoundError:
            logger.warning('File Not Found on Data Directory. Creating a new one from scratch')

    data = list()

    handler = open(path).read()
    soup = Soup(handler, "lxml")

    reviews = soup.body.reviews

    for body_child in reviews:
        if isinstance(body_child, Tag):
            for body_child_2 in body_child.sentences:
                if isinstance(body_child_2, Tag):
                    if body_child_2.opinions:
                        opinion = body_child_2.opinions.opinion
                        # keeping only reviews that have a polarity
                        if opinion:
                            sentence = body_child_2.text.strip()
                            polarity = opinion.attrs.get('polarity')
                            data.append({'text': sentence, 'polarity': polarity})

    extracted_data = pd.DataFrame(data)
    extracted_data = extracted_data[extracted_data['polarity'] != 'neutral']
    if save_data:
        logger.info('Saving etracted reviews metadata from file: {}'.format(file_type))
        x = path.split('.')[-1]
        outfile = re.sub(x, 'csv', path)
        extracted_data.to_csv(outfile, encoding='utf-8', index=False)

    return extracted_data


def get_df_stratified_split_in_train_validation(data,
                                                label,
                                                validation_size,
                                                random_state=5):
    """

    :param data: pandas dataframe with the data
    :param label: str. the label of the stratified split
    :param validation_size: float. the size of validation
    :param random_state: int. seed for reproducibility
    :return: dictionary
    """

    data.reset_index(drop=True, inplace=True)
    X = data.drop(label, axis=1).copy()
    y = data[label].copy()

    train_validation_sss = StratifiedShuffleSplit(test_size=validation_size,
                                                  random_state=random_state)

    x_train, x_val, y_train, y_val = None, None, None, None

    for train_index, validation_index in train_validation_sss.split(X, y):
        x_train, x_val = X.iloc[train_index], X.iloc[validation_index]
        y_train, y_val = y.iloc[train_index], y.iloc[validation_index]

    proportions = lambda x: sorted([(i, round(Counter(x)[i] / float(len(x)) * 100.0, 3)) for i in Counter(x)],
                                   key=lambda x: x[1])

    data_size = lambda x: round(len(x) / len(data), 3)

    logger.info(60 * '*')

    logger.info('Train Dataset real Size: {}'.format(round(len(x_train) / len(data), 3)))

    logger.info('Validation Dataset Requested Size: {}'.format(validation_size))
    logger.info('Validation Dataset Real Size: {}'.format(data_size(x_val)))

    logger.info('Train Dataset shape: {}'.format(x_train.shape))
    logger.info('Train Dataset Label Proportion: ')
    for t in proportions(y_train):
        logger.info(t)

    logger.info('Validation Dataset shape: {}'.format(x_val.shape))
    logger.info('Validation Dataset Label Proportion: ')
    for t in proportions(y_val):
        logger.info(t)

    df_train = x_train.copy()
    df_train[label] = y_train

    df_validation = x_val.copy()
    df_validation[label] = y_val

    result = {
        'x_train': x_train,
        'x_validation': x_val,
        'y_train': y_train,
        'y_validation': y_val,
        'df_train': df_train,
        'df_validation': df_validation}

    return result


if __name__ == "__main__":
    train_data = parse_reviews(load_data=False, save_data=False, file_type='train')
    print(train_data.head())

    calculate_label_ratio(train_data['polarity'])
