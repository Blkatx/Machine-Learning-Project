from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Tree Classifier Model
# - Fits the model without hyperparameter tuning
# - Does not use stratification
# - No cross-validation applied


def train_tree_classifier(X_train, X_test, y_train, y_test):
    """
    Train a Decision Tree classifier without hyperparameter tuning or cross-validation.

    Parameters:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target labels.
        X_test (array-like): Test feature data.
        y_test (array-like): Test target labels.

    Returns:
        float: Accuracy score on the test set.
    """
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with open("outputs/tree_classifier_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    return acc
