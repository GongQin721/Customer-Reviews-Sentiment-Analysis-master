from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tea.evaluation import create_clf_report, create_benchmark_plot, plot_roc_binary
from tea.features import *
from tea.load_data import parse_reviews, get_df_stratified_split_in_train_validation

if __name__ == "__main__":
    # loading data (train and test)
    train_data = parse_reviews(load_data=False, file_type='train')
    test_data = parse_reviews(load_data=False, file_type='test')

    split_metadata_dict = get_df_stratified_split_in_train_validation(data=train_data,
                                                                      label='polarity',
                                                                      validation_size=0.2,
                                                                      random_state=5)

    X_train = split_metadata_dict['x_train']
    X_val = split_metadata_dict['x_validation']

    X_test = test_data.drop(['polarity'], axis=1)

    y_train = split_metadata_dict['y_train']
    y_val = split_metadata_dict['y_validation']

    y_test = test_data['polarity']

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    # Lemmatizing all X's.
    X_train_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_train))

    X_val_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_val))

    X_train_val_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_train_val))

    X_test_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_test))

    # Setting the best vector based features with their parameters
    vect_based_features = Pipeline([('extract', TextColumnExtractor(column='text')),
                                    ('contractions', ContractionsExpander()),
                                    ('vect', CountVectorizer(binary=True,
                                                             min_df=0.01,
                                                             max_features=None,
                                                             ngram_range=(1, 1),
                                                             stop_words=None)),
                                    ('tfidf', TfidfTransformer(norm='l2', use_idf=False)),
                                    ('to_dense', DenseTransformer())])

    # Setting the best parameters for the user based features
    user_based_features = FeatureUnion(transformer_list=[
        ('text_length', TextLengthExtractor(col_name='text', reshape=True)),
        ('avg_token_length', WordLengthMetricsExtractor(col_name='text', metric='avg', split_type='thorough')),
        ('std_token_length', WordLengthMetricsExtractor(col_name='text', metric='std', split_type='thorough')),
        ('contains_spc', ContainsSpecialCharactersExtractor(col_name='text')),
        ('n_tokens', NumberOfTokensCalculator(col_name='text')),
        ('contains_dots_bool', ContainsSequentialChars(col_name='text', pattern='..')),
        ('contains_excl_bool', ContainsSequentialChars(col_name='text', pattern='!!')),
        ('sentiment_positive', HasSentimentWordsExtractor(col_name='text', sentiment='positive', count_type='counts')),
        ('sentiment_negative', HasSentimentWordsExtractor(col_name='text', sentiment='negative', count_type='boolean')),
        ('contains_uppercase', ContainsUppercaseWords(col_name='text', how='count'))])

    # setting the best classifier.
    best_clf = LogisticRegression(C=0.1, penalty='l1')

    # setting the final pipeline.
    final_pipeline = Pipeline([
        ('features', FeatureUnion(transformer_list=[
            ('vect_based_feat', vect_based_features),
            ('user_based_feat', user_based_features)])),
        ('scaling', StandardScaler()),
        ('clf', best_clf)])

    # we also need the pipeline without the clf in order to create a benchmark plotting.
    final_pipeline_without_clf = Pipeline([
        ('features', FeatureUnion(transformer_list=[
            ('vect_based_feat', vect_based_features),
            ('user_based_feat', user_based_features)])),
        ('scaling', StandardScaler())])

    X_train_benchmark = final_pipeline_without_clf.fit_transform(X_train_lemmatized)
    X_val_benchmark = final_pipeline_without_clf.transform(X_val_lemmatized)
    benchmark_clf = LogisticRegression(C=0.1)

    create_benchmark_plot(train_X=X_train_benchmark,
                          train_y=y_train,
                          test_X=X_val_benchmark,
                          test_y=y_val,
                          clf=benchmark_clf,
                          min_y_lim=0)

    # transforming labels in binary encoding (needed for plotting)
    le = LabelEncoder()
    y_train_val_enc = le.fit_transform(y_train_val)
    y_test_enc = le.transform(y_test)

    # fitting the model in all the training data (train and dev)
    fitted_model = final_pipeline.fit(X=X_train_val_lemmatized, y=y_train_val)
    y_test_pred = fitted_model.predict(X_test_lemmatized)

    # plot_roc_binary(X=X_train_val_lemmatized, y=y_train_val_enc, fitted_clf=fitted_model)
    # plot_roc_binary(X=X_test_lemmatized, y=y_test_enc, fitted_clf=fitted_model)

    create_clf_report(y_true=y_test,
                      y_pred=y_test_pred,
                      classes=fitted_model.classes_)
