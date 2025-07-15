from sklearn.model_selection import train_test_split
import pandas as pd

from preprocess import load_and_clean_data
from models.tree_classifier import train_tree_classifier
from models.tree_classifier_gs import train_tree_classifier_with_grid_search
from models.tree_classifier_random import train_tree_classifier_with_random_search
from models.random_forest_classifier import train_random_forest_with_random_search


if __name__ == "__main__":
    scaled_encoded_data, target_column = load_and_clean_data("personality_dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(scaled_encoded_data.drop(columns=[target_column]), scaled_encoded_data[target_column], test_size=0.2, random_state=42, stratify=scaled_encoded_data[target_column])

    accuracy_tree = train_tree_classifier(X_train, X_test, y_train, y_test)
    print(f"Accuracy of the Tree Classifier: {accuracy_tree:.4f}")

    accuracy_tree_gs, best_params, best_score = train_tree_classifier_with_grid_search(X_train, X_test, y_train, y_test)
    print(f"Accuracy of the Tree Classifier with Grid Search: {accuracy_tree_gs:.4f}")
    print(f"Best Parameters: {best_params}, Best Cross-Validation Score: {best_score:.4f}")

    accuracy_tree_random, best_params_random, best_score_random = train_tree_classifier_with_random_search(X_train, X_test, y_train, y_test)
    print(f"Accuracy of the Tree Classifier with Random Search: {accuracy_tree_random:.4f}")
    print(f"Best Parameters (Random Search): {best_params_random}, Best Cross-Validation Score: {best_score_random:.4f}")

    accuracy_rf_random, best_params_rf, best_score_rf = train_random_forest_with_random_search(X_train, y_train, X_test, y_test)
    print(f"Accuracy of the Random Forest Classifier with Random Search: {accuracy_rf_random:.4f}")
    print(f"Best Parameters (Random Forest): {best_params_rf}, Best Cross-Validation Score: {best_score_rf:.4f}")

    results = [
        {
            "model": "Decision Tree Classifier",
            "accuracy": accuracy_tree,
            "best_params": None,
            "cv_score": None
        },
        {
            "model": "Decision Tree (Grid Search)",
            "accuracy": accuracy_tree_gs,
            "best_params": best_params,
            "cv_score": best_score
        },
        {
            "model": "Decision Tree (Random Search)",
            "accuracy": accuracy_tree_random,
            "best_params": best_params_random,
            "cv_score": best_score_random
        },
        {
            "model": "Random Forest (Random Search)",
            "accuracy": accuracy_rf_random,
            "best_params": best_params_rf,
            "cv_score": best_score_rf
        }
    ]

    df = pd.DataFrame(results)
    df.to_csv("results/model_results.csv", index=False)
    print("Model results saved to model_results.csv")
