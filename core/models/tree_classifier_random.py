from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from models import utils

# Tree Classifier Model with Hyperparameter Tuning (RandomizedSearchCV)
# - Uses RandomizedSearchCV for hyperparameter tuning
# - Applies stratified cross-validation
# - Returns best accuracy, best parameters, and best cross-validation score

def train_tree_classifier_with_random_search(X_train, X_test, y_train, y_test, random_state=17, verbose=1, n_jobs=-1, n_iter=100):
    params = utils.parameters["DecisionTreeClassifierRandomSearch"]
    clf_random = DecisionTreeClassifier(random_state=random_state)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    random_search = RandomizedSearchCV(
        estimator=clf_random,
        param_distributions=params,
        cv=skf,
        verbose=verbose,
        n_jobs=n_jobs,
        n_iter=n_iter
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    y_pred = random_search.predict(X_test)
    best_acc = accuracy_score(y_test, y_pred)
    return best_acc, best_params, best_score