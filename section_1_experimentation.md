# Section 1. Experimentation

## Data Management

### Read and write a Delta table

#### Read Delta table

```Python
# Read table from absolute path
df = spark.read.format('delta').load('dbfs:/user/hive/warehouse/schema_name.db/table_name')
df = spark.read.load('/user/hive/warehouse/test_db.db/sample_1')

# Read table from metastore using relative namespace
df = spark.table('schema_name.table_name')
df = spark.read.table('schema_name.table_name')
df = spark.read.format('delta').table('schema_name.table_name')

# Read with Spark SQL
df = spark.sql('SELECT * FROM schema_name.table_name')

# Read with DeltaTable API
from delta.tables import DeltaTable
df = DeltaTable.forPath(spark, "/user/hive/warehouse/test_db.db/sample_1").toDF()
```

#### Write Delta table

```Python
# use absolute path
df.write.format("delta").mode("overwrite").save('dbfs:/user/hive/warehouse/schema_name.db/table_name...')

# use relative path
df.write.format("delta").mode("overwrite").saveAsTable("schema_name.table_name")
df.write.format("delta").saveAsTable("schema_name.table_name")

df.write.saveAsTable('schema_name.table_name')
df.write.mode("append").saveAsTable('schema_name.table_name')

# Additional parameters
(
    df.write.format("delta")
    .mode("overwrite")
    .partitionBy(colName)
    .option("overwriteSchema", "true")
    .save(deltaPath)
)
```

### View Delta table history and load a previous version of a Delta table

```sql
%sql
DESCRIBE HISTORY default.table_name
DESCRIBE HISTORY '/user/hive/warehouse/test_db.db/sample_1'

SELECT * FROM table_name VERSION AS OF 3

RESTORE TABLE table_name TO VERSION AS OF 8 
```

```Python
# Display history
deltaTable = DeltaTable.forName(spark, 'schema_name.table_name')
display(deltaTable.history())

# Load previous version
df = spark.read.format("delta").option("versionAsOf", 0).load(deltaPath)
df = spark.read.format("delta").option("timestampAsOf", timeStampString).load(deltaPath)
```

### Create, overwrite, merge, and read Feature Store tables in machine learning workflows

#### Create feature table
```Python
# Workspace Feature Store
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

# Feature Engineering in Unity Catalog
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
fs = FeatureEngineeringClient()

# Create table from dataframe
fs.create_table(
    name=table_name,
    primary_keys=["id"],
    df=airbnb_df,
    partition_columns=["neighbourhood"], # or 'date'
    description="Original Airbnb data"
)

# Create empty feature table with schema and populate it later
fs.create_table(
    name=table_name,
    primary_keys=["index"],
    schema=airbnb_df.schema,
    description="Original Airbnb data",
    # timestamp_keys=['timestamp'],
    # tags={}
)

fs.write_table(
    name=table_name,
    df=airbnb_df,
    mode="overwrite" # mode='append' to merge records or columns
)

# Read metadata for feature table
fs.get_table(table_name).description
fs.get_table(table_name).path_data_sources

# Register existing table
fs.register_table(
    delta_table: str,
    primary_keys: Union[str, List[str]],
    timestamp_keys: Union[str, List[str], None] = None,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
)

# Read feature table to dataframe
fs.read_table(name: str, **kwargs) → pyspark.sql.dataframe.DataFrame

```

#### Use feature table to create training data

```Python
from databricks.feature_store import FeatureLookup, FeatureStoreClient

# The model training uses two features from the 'customer_features' feature table and
# a single feature from 'product_features'
feature_lookups = [
    FeatureLookup(
      table_name='recommender_system.customer_features',
      feature_names=['total_purchases_30d', 'total_purchases_7d'],
      lookup_key='customer_id'
    ),
    FeatureLookup(
      table_name='recommender_system.product_features',
      feature_names=['category'],
      lookup_key='product_id'
    )
  ]

fs = FeatureStoreClient()

# Create a training set using training DataFrame and features from Feature Store
# The training DataFrame must contain all lookup keys from the set of feature lookups,
# in this case 'customer_id' and 'product_id'. It must also contain all labels used
# for training, in this case 'rating'.
training_set = fs.create_training_set(
  df=training_df, # can contain additional columns in addition to feature columns
  feature_lookups=feature_lookups,
  label='rating',
  exclude_columns=['customer_id', 'product_id']
)

training_df = training_set.load_df()
```
#### Log model with Feature Store metadata

```Python
fs.log_model(
    model=model,
    artifact_path="recommendation_model",
    flavor=mlflow.sklearn,
    training_set=training_set,
    registered_model_name="recommendation_model"
)
```

#### Use Feature Store logged model for batch inference

```Python
predictions = fs.score_batch(
    model_uri=model_uri,
    df=batch_df
)
```

## Experiment Tracking

### Manually log parameters, models, and evaluation metrics using MLflow

#### Create an experiment
```Python
mlflow.create_experiment(
    name: str,
    artifact_location: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None) → str
# Returns String ID of the created experiment.

# Get experiment metadata
experiment = mlflow.get_experiment(experiment_id)
print(f"Name: {experiment.name}")
print(f"Experiment_id: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Tags: {experiment.tags}")
print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
print(f"Creation timestamp: {experiment.creation_time}")
```

#### Use existing experiment (creates new if not exists)
```Python
mlflow.set_experiment(
    experiment_name: Optional[str] = None,
    experiment_id: Optional[str] = None) → Experiment
```

#### Start a new run
```Python
# Starts a new run or returns currently active run if exists
mlflow.start_run(
    run_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    run_name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    log_system_metrics: Optional[bool] = None) → ActiveRun

# Returns mlflow.ActiveRun object that acts as a context manager wrapping the run’s state.
```
**Run object structure**

```
mlflow.entities.Run
    .data (mlflow.entities.RunData)
        .metrics (dict)
        .params (dict)
        .tags (dict)
    .info (mlflow.entities.RunInfo)
        .artifact_uri (str)
        .end_time
        .experiment_id
        .lifecycle_stage
        .run_id
        .run_name
        .start_time
        .status (mlflow.entities.RunStatus)
        .user_id
    .inputs (mlflow.entities.RunInputs)
        .dataset_inputs
```

#### Log parameters

```Python
mlflow.log_param(key: str, value: Any) → Any[source]
mlflow.log_params(params: Dict[str, Any], synchronous: bool = True
    ) → Optional[mlflow.utils.async_logging.run_operations.RunOperations]

# Example
with mflow.start_run():
    value = mlflow.log_param("learning_rate", 0.01)
    assert value == 0.01

# Log multiple params
params = {"learning_rate": 0.01, "n_estimators": 10}
with mlflow.start_run():
    mlflow.log_params(params)
```
#### Log metrics

```Python
mlflow.log_metric(
    key: str, value: float, step: Optional[int] = None, synchronous: bool = True
    ) → Optional[mlflow.utils.async_logging.run_operations.RunOperations]
mlflow.log_metrics(
    metrics: Dict[str, float], step: Optional[int] = None, synchronous: bool = True
    ) → Optional[mlflow.utils.async_logging.run_operations.RunOperations]

# Example
with mlflow.start_run():
    mlflow.log_metric("mse", 2500.00)

# Log a batch of metrics
metrics = {"mse": 2500.00, "rmse": 50.00}
with mlflow.start_run():
    mlflow.log_metrics(metrics)

```
Extra stuff:

`mlflow.log_input() `logs a single mlflow.data.dataset.Dataset object corresponding to the currently active run. You may also log a dataset context string and a dict of key-value tags.

`mlflow.set_tag()` sets a single key-value tag in the currently active run. The key and value are both strings. Use `mlflow.set_tags()` to set multiple tags at once.

`mlflow.log_artifact()` logs a local file or directory as an artifact, optionally taking an artifact_path to place it in within the run’s artifact URI. Run artifacts can be organized into directories, so you can place the artifact in a directory this way.

`mlflow.log_artifacts()` logs all the files in a given directory as artifacts, again taking an optional artifact_path.

`mlflow.get_artifact_uri()` returns the URI that artifacts from the current run should be logged to.


#### Log models

Manual model logging is performed using per-flavor `log_model` methods.

```Python
# Log Sklearn model
with mlflow.start_run():
    # Train a sklearn model on the iris dataset
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    clf = RandomForestClassifier(max_depth=7)
    clf.fit(X, y)
    # Take the first row of the training dataset as the model input example.
    input_example = X.iloc[[0]]
    # Log the model and register it as a new version in UC.
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        # The signature is automatically inferred from the input example and its predicted output.
        input_example=input_example,
        registered_model_name="prod.ml_team.iris_model",
    )
..............
# Log Transformers model
from transformers import MobileBertForQuestionAnswering, AutoTokenizer

architecture = "csarron/mobilebert-uncased-squad-v2"
tokenizer = AutoTokenizer.from_pretrained(architecture)
model = MobileBertForQuestionAnswering.from_pretrained(architecture)

with mlflow.start_run():
    components = {
        "model": model,
        "tokenizer": tokenizer,
    }
    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path="my_model",
    )
...............
# XGBoost log_model 
mlflow.xgboost.log_model(
    xgb_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: mlflow.models.signature.ModelSignature = None,
    input_example: Union[
        pandas.core.frame.DataFrame, numpy.ndarray, dict, list,
        csr_matrix, csc_matrix, str, bytes, tuple] = None,
    await_registration_for=300,
    pip_requirements=None,
    extra_pip_requirements=None,
    model_format='xgb',
    metadata=None,
    **kwargs)
```

### Programmatically access and use data, metadata, and models from MLflow experiments

#### Search experiments programmatically
```Python
mlflow.search_experiments(
    view_type: int = 1,
    max_results: Optional[int] = None,
    filter_string: Optional[str] = None,
    order_by: Optional[List[str]] = None
    ) → List[Experiment]

MlflowClient.search_experiments(
    view_type: int = 1,
    max_results: Optional[int] = 1000,
    filter_string: Optional[str] = None,
    order_by: Optional[List[str]] = None,
    page_token=None
    ) → PagedList[Experiment]

# EXAMPLES
# Search for experiments with name "a"
experiments = client.search_experiments(filter_string="name = 'a'")
assert_experiment_names_equal(experiments, ["a"])

# Search for experiments with name starting with "a"
experiments = client.search_experiments(filter_string="name LIKE 'a%'")
assert_experiment_names_equal(experiments, ["ab", "a"])

# Search for experiments with tag key "k" and value ending with "v" or "V"
experiments = client.search_experiments(filter_string="tags.k ILIKE '%v'")
assert_experiment_names_equal(experiments, ["bb", "ab"])

# Search for experiments with name ending with "b" and tag {"k": "v"}
experiments = client.search_experiments(filter_string="name LIKE '%b' AND tags.k = 'v'")
assert_experiment_names_equal(experiments, ["ab"])

# Sort experiments by name in ascending order
experiments = client.search_experiments(order_by=["name"])
assert_experiment_names_equal(experiments, ["a", "ab", "b", "bb"])

# Sort experiments by ID in descending order
experiments = client.search_experiments(order_by=["experiment_id DESC"])
assert_experiment_names_equal(experiments, ["bb", "ab", "b", "a"])
```

#### Search runs programmatically

`mlflow.search_runs`
```Python
mlflow.search_runs(
    experiment_ids: Optional[List[str]] = None,
    filter_string: str = '',
    run_view_type: int = 1,
    max_results: int = 100000,
    order_by: Optional[List[str]] = None,
    output_format: str = 'pandas',
    search_all_experiments: bool = False,
    experiment_names: Optional[List[str]] = None
    ) → Union[List[Run], pandas.DataFrame]

# EXAMPLES
experiment_name = "Social NLP Experiments"
experiment_id = mlflow.create_experiment(experiment_name)
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_metric("m", 1.55)
    mlflow.set_tag("s.release", "1.1.0-RC")
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_metric("m", 2.50)
    mlflow.set_tag("s.release", "1.2.0-GA")

# Search for all the runs in the experiment with the given experiment ID
df = mlflow.search_runs([experiment_id], order_by=["metrics.m DESC"])
print(df[["metrics.m", "tags.s.release", "run_id"]])
print("--")

# Search the experiment_id using a filter_string with tag
# that has a case insensitive pattern
filter_string = "tags.s.release ILIKE '%rc%'"
df = mlflow.search_runs([experiment_id], filter_string=filter_string)
print(df[["metrics.m", "tags.s.release", "run_id"]])
print("--")

# Search for all the runs in the experiment with the given experiment name
df = mlflow.search_runs(experiment_names=[experiment_name], order_by=["metrics.m DESC"])
print(df[["metrics.m", "tags.s.release", "run_id"]])
```
Use `MlflowClient.search_runs()`
```Python
# MLFlow Client
MlflowClient.search_runs(
    experiment_ids: List[str],
    filter_string: str = '',
    run_view_type: int = 1,
    max_results: int = 1000,
    order_by: Optional[List[str]] = None,
    page_token: Optional[str] = None
    ) → PagedList[Run]

# EXAMPLES
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

client = MlflowClient()

display(client.list_run_infos(experiment_id))

runs = client.search_runs(experiment_id, order_by=["metrics.accuracy DESC"])

# Identify the best active run from experiment ID 0 by accuracy
run = client.search_runs(
    experiment_ids="0",
    filter_string="",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.accuracy DESC"],
)[0]

# get all active runs from experiments IDs 3, 4, and 17 that used a CNN model with 10 layers and had a prediction accuracy of 94.5% or higher
query = "params.model = 'CNN' and params.layers = '10' and metrics.`prediction accuracy` >= 0.945"
runs = client.search_runs(
    experiment_ids=["3", "4", "17"],
    filter_string=query,
    run_view_type=ViewType.ACTIVE_ONLY,
)

# search all known experiments for any MLflow runs created using the Inception model architecture
all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
runs = mlflow.search_runs(
    experiment_ids=all_experiments,
    filter_string="params.model = 'Inception'",
    run_view_type=ViewType.ALL,
)
```
#### Access models from experiment / run

```Python
run = mlflow.last_active_run()
# or
run = MlflowClient().get_run(run_id)

model = mlflow.sklearn.load_model(f'runs:/{run.info.run_id}/sk_models')
```

## Advanced Experiment Tracking

### Perform MLflow experiment tracking workflows using model signatures and input examples

### Identify the requirements for tracking nested runs

### Describe the process of enabling autologging, including with the use of Hyperopt

### Log and view artifacts like SHAP plots, custom visualizations, feature data, images, and metadata