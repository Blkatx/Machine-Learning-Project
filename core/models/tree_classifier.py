from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Tree Classifier Model
# - Fits the model without hyperparameter tuning
# - Does not use stratification
# - No cross-validation applied

def train_tree_classifier(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc