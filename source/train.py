import duckdb
import xgboost
import pathlib
import mlflow
import dvc.api
import sklearn
import os
import numpy as np

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

params = dvc.api.params_show()
data_path = pathlib.Path(__file__).parents[1] / "data"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(experiment_name=params["experiment"]["name"])

data_frame = duckdb.read_parquet(
    str(data_path / "prepared/train.parquet")
).to_df()


object_columns = ["gender", "age_group", "inventory_id", "day_of_week", "hour", "seq_first", "seq_last", "seq_length"]

data_frame[object_columns] = data_frame[object_columns].astype(np.float16)

train, valid = sklearn.model_selection.train_test_split(
    data_frame,
    test_size=0.1,
    random_state=42,
    stratify=data_frame["clicked"],
)

train_pool = xgboost.DMatrix(
    train.drop(columns=["clicked"]),
    train["clicked"],
)

valid_pool = xgboost.DMatrix(
    valid.drop(columns=["clicked"]),
    valid["clicked"],
)

pos_ratio = train["clicked"].mean()
scale_pos_weight = (1 - pos_ratio) / pos_ratio

parameters = {
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'verbosity': 0,
    'seed': 42
}

with mlflow.start_run() as run:
    model = xgboost.train(
        params=parameters, dtrain=train_pool,
        early_stopping_rounds=100, evals=[(train_pool,'train'),(valid_pool,'eval')]
    )

    mlflow.xgboost.log_model(
        model,
        name=params["model"]["name"],
        input_example=train.head(),
    )

    mlflow.log_params(parameters)

    metrics = {
        "average_precision_score": sklearn.metrics.average_precision_score(
            valid["clicked"], model.predict(valid_pool)
        ),
        "logloss": sklearn.metrics.log_loss(
            valid["clicked"], model.predict(valid_pool)
        ),
    }

    mlflow.log_metrics(metrics)
    print(metrics)

    model_save_path = data_path / "models/models.json"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_save_path)
