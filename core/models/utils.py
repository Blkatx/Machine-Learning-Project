parameters = {
  "DecisionTreeClassifierGridSearch": {
    "max_depth": range(6, 9),
    "min_samples_leaf": range(6, 9),
    "max_features": range(6, 9),
    "criterion": ["gini", "entropy"],
  },
  "DecisionTreeClassifierRandomSearch": {
    "max_depth": range(2, 50),
    "min_samples_leaf": range(2, 50),
    "max_features": range(2, 50),
    "criterion": ["gini", "entropy"],
  },
  "RandomForestClassifierRandomSearch": {
    "n_estimators": range(2, 500),
    "max_depth": range(2, 50),
    "min_samples_leaf": range(2, 50),
    "max_features": range(2, 50),
    "criterion": ["gini", "entropy"],
  },
}
