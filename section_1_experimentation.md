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
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

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

### Programmatically access and use data, metadata, and models from MLflow experiments

## Advanced Experiment Tracking

### Perform MLflow experiment tracking workflows using model signatures and input examples

### Identify the requirements for tracking nested runs

### Describe the process of enabling autologging, including with the use of Hyperopt

### Log and view artifacts like SHAP plots, custom visualizations, feature data, images, and metadata