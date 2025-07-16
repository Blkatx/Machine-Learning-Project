from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from models import utils
import pickle

# Tree Classifier Model with Hyperparameter Tuning
# - Uses GridSearchCV for hyperparameter tuning
# - Applies stratified cross-validation
# - Returns best accuracy, best parameters, and best cross-validation score


def train_tree_classifier_with_grid_search(
    X_train, X_test, y_train, y_test, random_state=17, verbose=1, n_jobs=-1
):
    """
    Train a Decision Tree classifier with hyperparameter tuning using GridSearchCV.

    Parameters:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target labels.
        X_test (array-like): Test feature data.
        y_test (array-like): Test target labels.
        random_state (int, optional): Random seed for reproducibility. Default is 17.
        verbose (int, optional): Verbosity level for GridSearchCV. Default is 1.
        n_jobs (int, optional): Number of jobs to run in parallel. Default is -1 (all CPUs).

    Returns:
        tuple: (test accuracy, best hyperparameters, best cross-validation score)

    Notes:
        - Uses parameters from utils.parameters["DecisionTreeClassifierGridSearch"].
        - Employs StratifiedKFold to maintain class distribution across folds.
    """

    param_grid = utils.parameters["DecisionTreeClassifierGridSearch"]
    clf = DecisionTreeClassifier(random_state=random_state)

    # Stratified K-Fold cross-validation
    # - Ensures that each fold has the same proportion of classes as the entire dataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=clf, param_grid=param_grid, cv=skf, verbose=verbose, n_jobs=n_jobs
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    best_acc = accuracy_score(y_test, y_pred)

    # Save the trained model to a file
    with open("outputs/tree_classifier_grid_search_model.pkl", "wb") as f:
        pickle.dump(grid_search, f)

    return best_acc, best_params, best_score
