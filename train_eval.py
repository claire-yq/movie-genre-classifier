from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.multioutput import MultiOutputClassifier
import sklearn.metrics as metrics

def grid_search_dev(X_train_unigrams, X_dev_unigrams,
                    X_train_avgemb50, X_dev_avgemb50,
                    X_train_avgemb100, X_dev_avgemb100,
                    y_train_true, y_dev_true):

    dev_hamming_loss = []
    dev_f1 = []
    dev_precision = []
    dev_recall = []

    def train_and_get_dev_metrics(base_model, X_train_features, X_dev_features, config_name):
        clf = MultiOutputClassifier(base_model)
        clf.fit(X_train_features, y_train_true)
        y_dev_pred = clf.predict(X_dev_features)
        dev_hamming_loss.append((config_name, metrics.hamming_loss(y_dev_true, y_dev_pred)))
        dev_f1.append((config_name, metrics.f1_score(y_dev_true, y_dev_pred, average='weighted')))
        dev_precision.append((config_name, metrics.precision_score(y_dev_true, y_dev_pred, average='weighted')))
        dev_recall.append((config_name, metrics.recall_score(y_dev_true, y_dev_pred, average='weighted')))

    # Unigrams
    features = (X_train_unigrams, X_dev_unigrams)
    for alpha in [0.05, 0.1, 0.2]:
        base_model = MultinomialNB(alpha=alpha)
        train_and_get_dev_metrics(base_model, features[0], features[1],
                                  f"Unigrams NB {alpha}")
    for C in [0.5, 1.0, 2.0, 10.0]:
        base_model = LogisticRegression(C=C, max_iter=1000)
        train_and_get_dev_metrics(base_model, features[0], features[1], f"Unigrams LR {C}")

    # Embeddings
    for dim_count, features in [("50d", (X_train_avgemb50, X_dev_avgemb50)),
                                ("100d", (X_train_avgemb100, X_dev_avgemb100))]:
        train_and_get_dev_metrics(GaussianNB(), features[0], features[1],
                                  f"Embeddings {dim_count} NB")
        for C in [0.5, 1.0, 2.0, 10.0]:
            base_model = LogisticRegression(C=C, max_iter=1000)
            train_and_get_dev_metrics(base_model, features[0], features[1],
                                      f"Embeddings {dim_count} LR {C}")

    return dev_hamming_loss, dev_f1, dev_precision, dev_recall


def train_and_get_test_metrics(X_train_unigrams, X_test_unigrams,
                               y_train_true, y_test_true):
    test_hamming_loss = []
    test_f1 = []
    test_precision = []
    test_recall = []

    for alpha in [0.05, 0.1, 0.2]:
        clf = MultiOutputClassifier(MultinomialNB(alpha=alpha))
        config_name = f"Unigrams NB {alpha}"

        clf.fit(X_train_unigrams, y_train_true)
        y_test_pred = clf.predict(X_test_unigrams)
        test_hamming_loss.append((config_name, metrics.hamming_loss(y_test_true, y_test_pred)))
        test_f1.append((config_name, metrics.f1_score(y_test_true, y_test_pred, average='weighted')))
        test_precision.append((config_name, metrics.precision_score(y_test_true, y_test_pred, average='weighted')))
        test_recall.append((config_name, metrics.recall_score(y_test_true, y_test_pred, average='weighted')))

    return test_hamming_loss, test_f1, test_precision, test_recall