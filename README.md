# Machine Learning Project: Personality Prediction

## Overview
This project aims to predict personality traits using various machine learning models. The workflow includes data preprocessing, model training, hyperparameter tuning, evaluation, and automated CI/CD with GitHub Actions.

## Project Structure
```
personality_dataset.csv        # Main dataset
requirements.txt               # Python dependencies
core/
    main.py                    # Main script for training and evaluation
    preprocess.py              # Data loading and cleaning
    models/
        tree_classifier.py             # Basic Decision Tree
        tree_classifier_gs.py          # Decision Tree with Grid Search
        tree_classifier_random.py      # Decision Tree with Random Search
        random_forest_classifier.py    # Random Forest with Random Search
        utils.py                       # Utility functions and parameter grids
results/
    model_results.csv          # Model performance results
```

## Models Implemented
- **Decision Tree Classifier**: Basic model without hyperparameter tuning.
- **Decision Tree (Grid Search)**: Uses GridSearchCV for hyperparameter optimization.
- **Decision Tree (Random Search)**: Uses RandomizedSearchCV for hyperparameter optimization.
- **Random Forest (Random Search)**: Uses RandomizedSearchCV for hyperparameter optimization.

All models use stratified train/test splits. The tuned models use cross-validation for robust parameter selection.

## Evaluation
- **Accuracy**: All models report accuracy on the test set.
- **Results**: Model metrics and best parameters are saved to `results/model_results.csv`.

## Data Preprocessing
- The `preprocess.py` script loads and cleans the dataset, encodes categorical variables, and scales features as needed.

## How to Run
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```
   python core/main.py
   ```
3. Check the `results/` folder for outputs.

## GitHub Actions CI/CD

Automated workflows are set up in `.github/workflows/basic_workflow.yaml`:
- **Python version matrix**: Runs on Python 3.13 (add more versions as needed).
- **Dependency installation**: Installs all requirements from `requirements.txt`.
- **Run training script**: Executes `python core/main.py` to train models and generate outputs.
- **Check outputs**: Lists the contents of the `outputs/` folder.
- **Upload model artifacts**: Uploads files from the `outputs/` folder as workflow artifacts.

This ensures your training pipeline runs automatically and model artifacts are saved for every push.

## Requirements
- Python 3.12+
- See `requirements.txt` for all dependencies (including scikit-learn, pandas, matplotlib, tabulate, etc.)

## License
MIT License

---
# Machine-Learning-Project