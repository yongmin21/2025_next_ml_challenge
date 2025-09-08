import duckdb
import catboost
import pathlib
import mlflow
import dvc.api
import sklearn
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

params = dvc.api.params_show()
data_path = pathlib.Path(__file__).parents[1] / "data"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(experiment_name=params["experiment"]["name"])

data_frame = duckdb.read_parquet(
    str(data_path / "prepared/train_under_sampled.parquet")
).to_df()

train, valid = sklearn.model_selection.train_test_split(
    data_frame,
    test_size=0.1,
    random_state=42,
    stratify=data_frame["clicked"],
)

train_pool = catboost.Pool(
    train.drop(columns=["clicked"]),
    train["clicked"],
)

valid_pool = catboost.Pool(
    valid.drop(columns=["clicked"]),
    valid["clicked"],
)

model = catboost.CatBoostClassifier(
    iterations=2000,
    learning_rate=0.02,
    loss_function="Logloss",
    verbose=False,
)

with mlflow.start_run() as run:
    model.fit(train_pool, eval_set=valid_pool)

    mlflow.catboost.log_model(
        model,
        name=params["model"]["name"],
        input_example=train.head(),
    )

    mlflow.log_params(params["model"])

    metrics = {
        "average_precision_score": sklearn.metrics.average_precision_score(
            valid["clicked"], model.predict_proba(valid_pool)[:, 1]
        ),
        "logloss": sklearn.metrics.log_loss(
            valid["clicked"], model.predict_proba(valid_pool)[:, 1]
        ),
    }

    mlflow.log_metrics(metrics)
    print(metrics)

    model_save_path = data_path / "models/models.onnx"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(
        model_save_path,
        format="onnx",
        export_parameters={
            "onnx_domain": "ai.catboost",
            "onnx_doc_string": "test model for BinaryClassification",
            "onnx_graph_name": "CatBoostModel_for_BinaryClassification",
        },
    )
