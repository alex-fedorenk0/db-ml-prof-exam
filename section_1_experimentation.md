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

#### Load experiment data by id and query runs as spark df
```Python
df = spark.read.format("mlflow-experiment").load("3270527066281272")
filtered_df = df.filter("metrics.loss < 0.01 AND params.learning_rate > '0.001'")
display(filtered_df)
```

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
#### Access models from run

```Python
run = mlflow.last_active_run()
# or
run = MlflowClient().get_run(run_id)

model = mlflow.sklearn.load_model(f'runs:/{run.info.run_id}/sk_models')
```

## Advanced Experiment Tracking

### Perform MLflow experiment tracking workflows using model signatures and input examples

#### Log model with input example
```Python
# Log column-based input example
input_example = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}
mlflow.sklearn.log_model(..., input_example=input_example)

# Log tensor-based input example
input_example = np.array(
    [
        [[0, 0, 0, 0], [0, 134, 25, 56], [253, 242, 195, 6], [0, 93, 82, 82]],
        [[0, 23, 46, 0], [33, 13, 36, 166], [76, 75, 0, 255], [33, 44, 11, 82]],
    ],
    dtype=np.uint8,
)
mlflow.tensorflow.log_model(..., input_example=input_example)
```

#### Log model with signature

##### Signature automatically inferred from input example
```Python
clf.fit(iris_train, iris.target)
# Take the first row of the training dataset as the model input example.
input_example = iris_train.iloc[[0]]
# The signature is automatically inferred from the input example and its predicted output.
mlflow.sklearn.log_model(clf, "iris_rf", input_example=input_example)
```

##### Signature explicitly constructed or inferrred
```Python
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

# Option 1: Manually construct the signature object
input_schema = Schema(
    [
        ColSpec("double", "sepal length (cm)"),
        ColSpec("double", "sepal width (cm)"),
        ColSpec("double", "petal length (cm)"),
        ColSpec("double", "petal width (cm)"),
    ]
)
output_schema = Schema([ColSpec("long")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Option 2: Infer the signature
signature = infer_signature(iris_train, clf.predict(iris_train))

with mlflow.start_run():
    mlflow.sklearn.log_model(clf, "iris_rf", signature=signature)

```

##### Set signature on a logged model
```Python
import mlflow
from mlflow.models.model import get_model_info
from mlflow.models import infer_signature, set_signature

# load the logged model
model_uri = f"runs:/{run.info.run_id}/iris_rf"
model = mlflow.pyfunc.load_model(model_uri)

# construct the model signature from test dataset
X_test, _ = datasets.load_iris(return_X_y=True, as_frame=True)
signature = infer_signature(X_test, model.predict(X_test))

# set the signature for the logged model
set_signature(model_uri, signature)

# now when you load the model again, it will have the desired signature
assert get_model_info(model_uri).signature == signature

```

### Identify the requirements for tracking nested runs

**What are Parent and Child Runs?**
At its core, MLflow allows users to track experiments, which are essentially named groups of runs. A “run” in this context refers to a single execution of a model training event, where you can log parameters, metrics, tags, and artifacts associated with the training process. The concept of Parent and Child Runs introduces a hierarchical structure to these runs.

Imagine a scenario where you’re testing a deep learning model with different architectures. Each architecture can be considered a parent run, and every iteration of hyperparameter tuning for that architecture becomes a child run nested under its respective parent.

**Benefits**

*Organizational Clarity*: By using Parent and Child Runs, you can easily group related runs together. For instance, if you’re running a hyperparameter search using a Bayesian approach on a particular model architecture, every iteration can be logged as a child run, while the overarching Bayesian optimization process can be the parent run.

*Enhanced Traceability*: When working on large projects with a broad product hierarchy, child runs can represent individual products or variants, making it straightforward to trace back results, metrics, or artifacts to their specific run.

*Scalability*: As your experiments grow in number and complexity, having a nested structure ensures that your tracking remains scalable. It’s much easier to navigate through a structured hierarchy than a flat list of hundreds or thousands of runs.

*Improved Collaboration*: For teams, this approach ensures that members can easily understand the structure and flow of experiments conducted by their peers, promoting collaboration and knowledge sharing.

**Relationship between Experiments, Parent Runs, and Child Runs**

*Experiments*: Consider experiments as the topmost layer. They are named entities under which all related runs reside. For instance, an experiment named “Deep Learning Architectures” might contain runs related to various architectures you’re testing.

*Parent Runs*: Within an experiment, a parent run represents a significant segment or phase of your workflow. Taking the earlier example, each specific architecture (like CNN, RNN, or Transformer) can be a parent run.

*Child Runs*: Nested within parent runs are child runs. These are iterations or variations within the scope of their parent. For a CNN parent run, different sets of hyperparameters or slight architectural tweaks can each be a child run.

```Python
# Create nested runs
experiment_id = mlflow.create_experiment("experiment1")
with mlflow.start_run(
    run_name="PARENT_RUN",
    experiment_id=experiment_id,
    tags={"version": "v1", "priority": "P1"},
    description="parent",
) as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(
        run_name="CHILD_RUN",
        experiment_id=experiment_id,
        description="child",
        nested=True,
    ) as child_run:
        mlflow.log_param("child", "yes")
```


### Describe the process of enabling autologging, including with the use of Hyperopt

Automatic logging allows you to log metrics, parameters, and models without the need for explicit log statements.
There are two ways to use autologging:

- Call `mlflow.autolog()` before your training code. This will enable autologging for each supported library you have installed as soon as you import it.
- Use library-specific autolog calls for each library you use in your code.

You can access the most recent autolog run through the `mlflow.last_active_run()` function.

#### The default configuration for the mlflow.autolog() call is:
```Python
mlflow.autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=True,
    disable_for_unsupported_versions=True,
    silent=True
)
```


#### Sklearn autolog example
```Python
from pprint import pprint
import numpy as np
from sklearn.linear_model import LinearRegression
import mlflow
from mlflow import MlflowClient


def fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


# enable autologging
mlflow.sklearn.autolog()

# prepare training data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# train a model
model = LinearRegression()
with mlflow.start_run() as run:
    model.fit(X, y)

# fetch logged data
params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)

pprint(params)
# {'copy_X': 'True',
#  'fit_intercept': 'True',
#  'n_jobs': 'None',
#  'normalize': 'False'}

pprint(metrics)
# {'training_score': 1.0,
#  'training_mean_absolute_error': 2.220446049250313e-16,
#  'training_mean_squared_error': 1.9721522630525295e-31,
#  'training_r2_score': 1.0,
#  'training_root_mean_squared_error': 4.440892098500626e-16}

pprint(tags)
# {'estimator_class': 'sklearn.linear_model._base.LinearRegression',
#  'estimator_name': 'LinearRegression'}

pprint(artifacts)
# ['model/MLmodel', 'model/conda.yaml', 'model/model.pkl']
```

#### Databricks autologging

Databricks Autologging is a no-code solution that extends MLflow automatic logging to deliver automatic experiment tracking for machine learning training sessions on Databricks.

With Databricks Autologging, model parameters, metrics, files, and lineage information are automatically captured when you train models from a variety of popular machine learning libraries. Training sessions are recorded as MLflow tracking runs. Model files are also tracked so you can easily log them to the MLflow Model Registry and deploy them for real-time scoring with Model Serving.

#### Hyperparameter tuning with HyperOpt

```Python
from hyperopt import SparkTrials
spark_trials = SparkTrials()
 
def objective(C):
    clf = SVC(C=C)
    accuracy = cross_val_score(clf, X, y).mean()
    return {'loss': -accuracy, 'status': STATUS_OK}

with mlflow.start_run():
  argmin = fmin(
    fn=objective,
    space=search_space,
    algo=algo,
    max_evals=16,
    trials=spark_trials)
```

##### SparkTrials and MLflow

Databricks Runtime ML supports logging to MLflow from workers. You can add custom logging code in the objective function you pass to Hyperopt.

SparkTrials logs tuning results as nested MLflow runs as follows:

Main or parent run: The call to `fmin()` is logged as the main run. If there is an active run, SparkTrials logs to this active run and does not end the run when `fmin()` returns. If there is no active run, SparkTrials creates a new run, logs to it, and ends the run before `fmin()` returns.

Child runs: Each hyperparameter setting tested (a “trial”) is logged as a child run under the main run. MLflow log records from workers are also stored under the corresponding child runs.

When calling `fmin()`, Databricks recommends active MLflow run management; that is, wrap the call to `fmin()` inside a with mlflow.start_run(): statement. This ensures that each `fmin()` call is logged to a separate MLflow main run, and makes it easier to log extra tags, parameters, or metrics to that run.

When using Hyperopt with MLlib and other distributed training algorithms, do not pass a `trials` argument to `fmin()`. When you do not include the `trials` argument, Hyperopt uses the default `Trials` class, which runs on the cluster driver. Hyperopt needs to evaluate each trial on the driver node so that each trial can initiate distributed training jobs.

Do not use the `SparkTrials` class with `MLlib`. `SparkTrials` is designed to distribute trials for algorithms that are not themselves distributed. MLlib uses distributed computing already and is not compatible with `SparkTrials`.


### Log and view artifacts like SHAP plots, custom visualizations, feature data, images, and metadata

#### Log artifacts
```Python
# Log a local file or directory as an artifact of the currently active run
mlflow.log_artifact(
    local_path: str,
    artifact_path: Optional[str] = None)

# Log all the contents of a local directory as artifacts of the run
mlflow.log_artifacts(
    local_dir: str,
    artifact_path: Optional[str] = None) 

# Log a figure as an artifact. The following figure objects are supported:
# - matplotlib.figure.Figure
# - plotly.graph_objects.Figure
mlflow.log_figure(
    figure: Union[matplotlib.figure.Figure, plotly.graph_objects.Figure],
    artifact_file: str,
    *, save_kwargs: Optional[Dict[str, Any]] = None)

# Log an image as an artifact. The following image objects are supported:
# numpy.ndarray, PIL.Image.Image
mlflow.log_image(
    image: Union[numpy.ndarray, PIL.Image.Image],
    artifact_file: str)
```

#### Load artifacts
```Python
# Download an artifact file or directory to a local directory.
mlflow.artifacts.download_artifacts(
    artifact_uri: Optional[str] = None,
    run_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
    dst_path: Optional[str] = None,
    tracking_uri: Optional[str] = None) 

# Loads artifact contents as a PIL.Image.Image object
mlflow.artifacts.load_image(artifact_uri: str)    

```
