from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from models import utils
import pickle

# Random Forest Classifier Model with Hyperparameter Tuning
# - Uses RandomizedSearchCV for hyperparameter tuning
# - Applies stratified cross-validation
# - Returns best accuracy, best parameters, and best cross-validation score


def train_random_forest_with_random_search(
    X_train, y_train, X_test, y_test, random_state=17, verbose=1, n_jobs=-1, n_iter=50
):
    """
    Train a Random Forest classifier with hyperparameter tuning using RandomizedSearchCV.

    Parameters:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target labels.
        X_test (array-like): Test feature data.
        y_test (array-like): Test target labels.
        random_state (int, optional): Random seed for reproducibility. Default is 17.
        verbose (int, optional): Verbosity level for RandomizedSearchCV. Default is 1.
        n_jobs (int, optional): Number of jobs to run in parallel. Default is -1 (all CPUs).
        n_iter (int, optional): Number of parameter settings sampled in RandomizedSearchCV. Default is 50.

    Returns:
        tuple: (test accuracy, best hyperparameters, best cross-validation score)
    """
    params = utils.parameters["RandomForestClassifierRandomSearch"]
    random_forest_clf = RandomForestClassifier(random_state=random_state)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    forest_search = RandomizedSearchCV(
        estimator=random_forest_clf,
        param_distributions=params,
        cv=skf,
        verbose=verbose,
        n_jobs=n_jobs,
        n_iter=n_iter,
    )
    forest_search.fit(X_train, y_train)
    best_params = forest_search.best_params_
    best_score = forest_search.best_score_
    y_pred = forest_search.predict(X_test)
    forest_search_acc = accuracy_score(y_test, y_pred)

    # Save the trained model to a file
    with open("outputs/random_forest_random_search_model.pkl", "wb") as f:
        pickle.dump(forest_search, f)

    return forest_search_acc, best_params, best_score
