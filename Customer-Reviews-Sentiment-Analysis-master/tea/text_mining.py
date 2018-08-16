import re
import unicodedata

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.base import BaseEstimator, TransformerMixin

from tea import CONTRACTION_MAP
from tea import setup_logger

logger = setup_logger(__name__)
# spacy_nlp = spacy.load('en', parse=False, tag=False, entity=False)
tokenizer = ToktokTokenizer()

STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.remove('no')
STOPWORDS.remove('not')


def remove_accents(input_str):
    """
    This method removes any accents from any string.

    :param input_str:
    :return:
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()


def tokenize_text(text, split_type='simple'):
    """

    :param text:
    :param split_type:
    :return:
    """
    assert split_type in ['simple', 'thorough']

    if text:
        if split_type == 'simple':
            return text.strip().split()

        else:
            label = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z])|(_|\.|-|!|'))", r' \1', text)
            name_list = re.sub('(_|\.|-|!|#)', "", label).split()
            # Getting rid of any digits and other leftovers attached to the words of the list
            name_list = [re.sub(r"(\d+)|('|,|\)|{|}|=|&|`)", ' ', x) for x in name_list]
            name_list = [item.strip().lower() for x in name_list for item in x.split()]
            return name_list

    else:
        return []


def extract_digits_from_text(text):
    """
    This function extracts any digits in a text
    :param text:
    :return:
    """
    return list(map(int, re.findall(r'\d+', text))) if text else []


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts a column that contains text and performs preprocessing."""

    def __init__(self,
                 col_name,
                 html_stripping=True,
                 contraction_expansion=True,
                 accented_char_removal=True,
                 text_lower_case=True,
                 text_lemmatization=True,
                 special_char_removal=True,
                 stopword_removal=True,
                 contractions_mapper=CONTRACTION_MAP,
                 stopwords=STOPWORDS):

        self.col_name = col_name
        self.html_stripping = html_stripping
        self.contraction_expansion = contraction_expansion
        self.accented_char_removal = accented_char_removal
        self.text_lower_case = text_lower_case
        self.text_lemmatization = text_lemmatization
        self.special_char_removal = special_char_removal
        self.stopword_removal = stopword_removal
        self.contractions_mapper = contractions_mapper
        self.stopwords = stopwords

    @staticmethod
    def strip_html_tags(text):
        """
        This method removes any html tags and keeps the clean text.
        :param text:
        :return:
        """
        soup = BeautifulSoup(text, "html.parser")

        stripped_text = soup.get_text()

        return stripped_text

    @staticmethod
    def remove_accented_chars(text):
        """
        This function converts accented characters\letters and standardizes them into ASCII characters.
        A simple example would be converting é to e.

        :param text: str.
        :return: str.
        """
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    @staticmethod
    def expand_contractions(text, contractions_m=CONTRACTION_MAP):
        """
        This function expands contractions for the english language. For example "I've" will become "I have".

        :param text:
        :param contractions_m: dict. A dict containing contracted words as keys, and their expanded text as values.
        :return:
        """

        contractions_pattern = re.compile('({})'.format('|'.join(contractions_m.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            """
            This sub function helps into expanding a given contraction
            :param contraction:
            :return:
            """
            match = contraction.group(0)
            first_char = match[0]

            expanded_contr = contractions_m.get(match) if contractions_m.get(match) else contractions_m.get(
                match.lower())
            expanded_contr = first_char + expanded_contr[1:]

            return expanded_contr

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)

        return expanded_text

    @staticmethod
    def remove_special_characters(text):
        """
        This method removes any special characters from text
        :param text:
        :return:
        """
        return re.sub('[^a-zA-z0-9\s]', '', text)

    def remove_stopwords(self, text, is_lower_case=False):
        """
        This function removes stopwords from text.
        :param text:
        :param is_lower_case:
        :return:
        """
        # tokenizing text
        tokens = tokenizer.tokenize(text)

        # stripping every token
        tokens = [token.strip() for token in tokens]

        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in self.stopwords]

        else:
            filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords]

        filtered_text = ' '.join(filtered_tokens)

        return filtered_text

    @staticmethod
    def lemmatize_text(text):
        """
        This method lemmatizes text
        :param text:
        :return:
        """
        text = spacy_nlp(text)

        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

        return text

    def normalize_doc(self, doc):
        """
        Ths method run a pipeline of text preprocessing. All parameters are given in the constructor of the class
        in order to be able to encapsulate it in an sklearn pipeline.
        :param doc:
        :return:
        """
        # strip HTML
        if self.html_stripping:
            doc = self.strip_html_tags(doc)

        # remove accented characters
        if self.accented_char_removal:
            doc = self.remove_accented_chars(doc)

        # expand contractions
        if self.contraction_expansion:
            doc = self.expand_contractions(doc)

        # lowercase the text
        if self.text_lower_case:
            doc = doc.lower()

        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)

        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)

        # lemmatize text
        if self.text_lemmatization:
            doc = self.lemmatize_text(doc)

        # remove special characters
        if self.special_char_removal:
            doc = self.remove_special_characters(doc)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        # remove stopwords
        if self.stopword_removal:
            doc = self.remove_stopwords(doc, is_lower_case=self.text_lower_case)

        return doc

    def transform(self, X, y=None):
        logger.info('Pre-processing text for "{}" Column'.format(self.col_name))
        return X[self.col_name].apply(self.normalize_doc)

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


if __name__ == "__main__":

    docs = ["<p>Héllo! Are you here? I just heard about <b>Python</b>!<br/>\r\n \n              It's an "
            "amazing language which can be used for Scripting, Web development,\r\n\r\n\n              Information "
            "Retrieval, Natural Language Processing, Machine Learning & Artificial Intelligence!\n\n              "
            "What are you waiting for? Go and get started.<br/> He's learning, she's learning, they've already\n\n\n  "
            "            got a headstart!</p>\n           "]

    df = pd.DataFrame(docs, columns=['text'])

    obj = TextPreprocessor(col_name='text', html_stripping=True)

    print(obj.transform(X=df)[0])
