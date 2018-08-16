import pandas as pd

from tea import DATA_DIR, setup_logger

logger = setup_logger(__name__)
from tqdm import tqdm


class WordEmbedding
    def __init__(self):
        pass

    @staticmethod
    def get_word_embeddings(dimension=200):
        """
        This method reads the
        :return: dict. with the vocabulary word and its word embedding vector in a list
        """
        assert dimension in [50, 100, 200, 300]

        t = 'glove.6B.{}d.txt'.format(dimension)

        logger.info('Loading Word Embeddings file: {}'.format(t))
        infile = "{}{}".format(DATA_DIR, t)

        with open(infile, 'rb') as in_file:
            text = in_file.read().decode("utf-8")

        word_embeddings = dict()
        for line in tqdm(text.split('\n'),
                         desc='Loading Embeddings for {} dimensions'.format(dimension),
                         unit=' Embeddings'):
            try:
                w_e_numbers = list(map(lambda x: float(x), line.split()[1:]))
                word_embeddings[line.split()[0]] = w_e_numbers
            except IndexError:
                pass

        return word_embeddings

    @staticmethod
    def get_word_embeddings_mean(dimension=50, save_data=False, load_data=True):
        """
        This method reads the embeddings, calculates the mean for every embedding and creates a mapper

        :param dimension:
        :param save_data:
        :param load_data:
        :return: dict. with the vocabulary word and its word embedding vector in a list
        """
        assert dimension in [50, 100, 200, 300]

        if load_data:
            try:
                t = 'glove.6B.{}d_mean.csv'.format(dimension)
                infile = "{}{}".format(DATA_DIR, t)
                logger.info('Loading file: {}'.format(infile))
                df = pd.read_csv(infile)

                return df.set_index(['word']).to_dict().get('mean_embedding')

            except FileNotFoundError:
                logger.warning('File Not Found in specified Directory. Creating a new one from scratch')

        t = 'glove.6B.{}d.txt'.format(dimension)

        logger.info('Loading Word Embeddings file: {}'.format(t))
        infile = "{}{}".format(DATA_DIR, t)

        with open(infile, 'rb') as in_file:
            text = in_file.read().decode("utf-8")

        word_embeddings = list()
        for line in tqdm(text.split('\n'),
                         desc='Loading Embeddings for {} dimensions'.format(dimension),
                         unit=' Embeddings'):
            try:
                w_e_numbers = list(map(lambda x: float(x), line.split()[1:]))

                word_embeddings.append((line.split()[0], np.mean(w_e_numbers)))
            except IndexError:
                pass

        mean_embeddings_df = pd.DataFrame(word_embeddings, columns=['word', 'mean_embedding'])

        if save_data:
            t = 'glove.6B.{}d_mean.csv'.format(dimension)
            outfile = "{}{}".format(DATA_DIR, t)
            mean_embeddings_df.to_csv(outfile, encoding='utf-8', index=False)

        return mean_embeddings_df.set_index(['word']).to_dict().get('mean_embedding')


if __name__ == '__main__':

    w_e = WordEmbedding.get_word_embeddings_mean(dimension=50, save_data=True, load_data=True)

    print('the: {}'.format(w_e['the']))
    print('a: {}'.format(w_e['a']))
    print('egg: {}'.format(w_e['egg']))

    import numpy as np

    print('the: {}'.format(np.mean(w_e['the'])))
