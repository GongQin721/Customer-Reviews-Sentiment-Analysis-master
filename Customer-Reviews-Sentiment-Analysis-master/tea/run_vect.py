from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from tea.features import *
from tea.load_data import parse_reviews
from tea.run_models import run_grid_search

if __name__ == "__main__":
    data = parse_reviews(load_data=False)

    X_train = data.drop(['polarity'], axis=1)
    y_train = data['polarity']

    X_train_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_train))

    vect_based_pipeline = Pipeline([('extract', TextColumnExtractor(column='text')),
                                    ('contractions', ContractionsExpander()),
                                    ('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer()),
                                    ('to_dense', DenseTransformer()),  # transforms sparse to dense
                                    ('scaling', StandardScaler()),
                                    # ('scaling', MinMaxScaler()),
                                    # ('pca', PCA()),
                                    # ('clf', SVC()),
                                    # ('clf', MultinomialNB())
                                    ('clf', LogisticRegression())
                                    # ('clf', KNeighborsClassifier())
                                    # ('clf', GradientBoostingClassifier())
                                    # ('clf', RandomForestClassifier())
                                    ])

    params = {
        'vect__min_df': (0.01, 0.05),
        'vect__max_features': (None, 1000, 2500, 5000),
        'vect__stop_words': (None, 'english'),
        'vect__binary': (True, False),
        'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams, bigrams, trigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'tfidf__smooth_idf': (True, False),  # do not use
        'tfidf__sublinear_tf': (True, False),  # do not use
        'clf__penalty': ('l1', 'l2'),  # Logistic
        # 'clf__kernel': ('rbf', 'linear'),  # SVM
        # 'clf__gamma': (0.1, 0.01, 0.001, 0.0001),  # SVM
        # 'clf__p': (1, 2),  # 1: mahnatan, 2: eucledian # k-NN
        # 'clf__n_neighbors': (3, 4, 5, 6, 7, 8),  # k-NN
        # 'clf__learning_rate': (0.1, 0.01, 0.001),  # Gradient Boosting
        # 'clf__n_estimators': (100, 300, 600),  # Gradient Boosting, Random Forest
        # 'clf__alpha': (0.5, 1.0),  # MultinomialNB
        # 'clf__max_depth': [10, 50, 100, None],  # Random Forest
    }

    grid_results = run_grid_search(X=X_train_lemmatized,
                                   y=y_train,
                                   pipeline=vect_based_pipeline,
                                   parameters=params,
                                   scoring='accuracy')
