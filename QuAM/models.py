import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from configurations import (
    CLASSIFICATION_TASK,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    CLASSIFICATION_MODEL_NAME,
    REGRESSION_MODEL_NAME,
    GAMMA_FIXED_OPTION,
    CLASS_WEIGHT_BALANCED_OPTION,
    CLASS_WEIGHT_NONE_OPTION,
    DROP_COLS,
)


def safe_int(value, default=None):
    # Tries to convert a value to an integer safely

    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=None):
    # Tries to convert a value to a float safely

    try:
        return float(value)
    except Exception:
        return default


def prepare_training_data(df, task):
    # Prepares the labelled dataset for training

    df = df.copy()

    # Selecting the target based on the chosen prediction task
    if task == CLASSIFICATION_TASK:
        if CLASSIFICATION_TARGET not in df.columns:
            raise ValueError(f"The labelled training dataset does not contain '{CLASSIFICATION_TARGET}'.")

        y = df[CLASSIFICATION_TARGET]
        target_name = CLASSIFICATION_TARGET

    else:
        if REGRESSION_TARGET not in df.columns:
            raise ValueError(f"The labelled training dataset does not contain '{REGRESSION_TARGET}'.")

        y = df[REGRESSION_TARGET]
        target_name = REGRESSION_TARGET

    # Removing target, id and leakage columns from the input features
    X = df.drop(columns=[col for col in DROP_COLS if col in df.columns],
                errors="ignore")

    # Only using rows that have a valid target value
    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    # Splitting columns by data type for preprocessing
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Numeric variables are imputed then scaled
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical variables are imputed then one-hot encoded
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combining preprocessing for both numeric and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return X, y, preprocessor, target_name


def prepare_query_features(df):
    # Prepares the query dataset for inference

    df = df.copy()

    # Dropping the same non-feature columns if they are present
    X = df.drop(columns=[col for col in DROP_COLS if col in df.columns],
                errors="ignore")

    return X


def build_model(task, params):
    # Builds either the classification or regression model

    random_state = safe_int(params.get("random_state"), 15288)

    if task == CLASSIFICATION_TASK:
        # Reading Kernel SVM parameters from the interface
        c_value = safe_float(params.get("c"), 1.0)
        gamma_mode = params.get("gamma_mode", "scale")
        class_weight_option = params.get("class_weight", CLASS_WEIGHT_BALANCED_OPTION)

        if gamma_mode == GAMMA_FIXED_OPTION:
            gamma_value = safe_float(params.get("gamma_fixed"), 0.1)
        else:
            gamma_value = gamma_mode

        if class_weight_option == CLASS_WEIGHT_BALANCED_OPTION:
            class_weight_value = "balanced"
        elif class_weight_option == CLASS_WEIGHT_NONE_OPTION:
            class_weight_value = None
        else:
            class_weight_value = "balanced"

        model = SVC(
            kernel="rbf",
            C=c_value,
            gamma=gamma_value,
            class_weight=class_weight_value,
            probability=True,
            random_state=random_state,
        )

    else:
        # Reading Decision Tree parameters from the interface
        max_depth = safe_int(params.get("max_depth"), 5)
        min_samples_leaf = safe_int(params.get("min_samples_leaf"), 20)

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    return model


def train_quam_model(train_df, query_df, task, params):
    # Trains the model and returns outputs needed by the interface

    X, y, preprocessor, target_name = prepare_training_data(train_df, task)

    test_size = safe_float(params.get("test_size"), 0.2)
    random_state = safe_int(params.get("random_state"), 15288)

    # Stratify is used only for classification to preserve class balance
    if task == CLASSIFICATION_TASK:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    model = build_model(task, params)

    # Pipeline keeps preprocessing and modelling together
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Query data is prepared after training if the user loaded it
    if query_df is not None:
        query_feature_df = prepare_query_features(query_df)
    else:
        query_feature_df = None

    # Calculating classification metrics
    if task == CLASSIFICATION_TASK:
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
        }

        output_text = (
            "MODEL TRAINED\n"
            f"Task: {task}\n"
            f"Model: {CLASSIFICATION_MODEL_NAME}\n"
            f"Target: {target_name}\n"
            f"Rows used: {len(X)}\n"
            f"Training rows: {len(X_train)}\n"
            f"Test rows: {len(X_test)}\n\n"
            "Classification metrics:\n"
            f"Accuracy : {acc:.4f}\n"
            f"Precision: {prec:.4f}\n"
            f"Recall   : {rec:.4f}\n"
            f"F1-score : {f1:.4f}\n\n"
            f"Confusion matrix:\n{cm}\n\n"
            f"Probability preview, first 10 test rows:\n{np.round(y_prob[:10], 4)}\n\n"
        )

    # Calculating regression metrics
    else:
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R^2": r2,
        }

        output_text = (
            "MODEL TRAINED\n"
            f"Task: {task}\n"
            f"Model: {REGRESSION_MODEL_NAME}\n"
            f"Target: {target_name}\n"
            f"Rows used: {len(X)}\n"
            f"Training rows: {len(X_train)}\n"
            f"Test rows: {len(X_test)}\n\n"
            "Regression metrics:\n"
            f"MAE : {mae:.4f}\n"
            f"RMSE: {rmse:.4f}\n"
            f"R^2 : {r2:.4f}\n\n"
        )

    # Returning everything the interface needs after training
    result = {
        "model": model,
        "pipeline": pipeline,
        "feature_df": X.copy(),
        "query_feature_df": query_feature_df,
        "target_name": target_name,
        "metrics": metrics,
        "output_text": output_text,
    }

    return result


def predict_single_row(pipeline, row_df, task):
    # Predicts the selected customer row

    # Classification case
    if task == CLASSIFICATION_TASK:
        pred_class = pipeline.predict(row_df)[0]
        pred_prob = pipeline.predict_proba(row_df)[0, 1]

        if pred_class == 1:
            label = "Delinquent"
        else:
            label = "Non-delinquent"

        result = {
            "pred_class": pred_class,
            "pred_prob": pred_prob,
            "label": label,
        }

    # For regression case
    else:
        pred_value = pipeline.predict(row_df)[0]

        result = {
            "pred_value": pred_value,
        }

    return result


def predict_all_delinquencies(pipeline, active_df):
    # Finds all rows predicted as delinquent

    pred_classes = pipeline.predict(active_df)
    pred_probs = pipeline.predict_proba(active_df)[:, 1]

    delinquency_results = []

    for row_index, pred_class, pred_prob in zip(active_df.index, pred_classes, pred_probs):
        if pred_class == 1:
            result = {
                "row_index": row_index,
                "pred_prob": pred_prob,
            }

            delinquency_results.append(result)

    delinquency_results = sorted(
        delinquency_results,
        key=lambda row: row["pred_prob"],
        reverse=True
    )

    return delinquency_results