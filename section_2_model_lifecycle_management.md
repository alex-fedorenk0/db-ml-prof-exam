# Section 2: Model Lifecycle Management

## Preprocessing Logic

### Describe an MLflow flavor and the benefits of using MLflow flavors



### Describe the advantages of using the pyfunc MLflow flavor

### Describe the process and benefits of including preprocessing logic and context in custom model classes and objects

## Model Management

### Describe the basic purpose and user interactions with Model Registry

The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model. It provides model lineage (which MLflow experiment and run produced the model), model versioning, model aliasing, model tagging, and annotations.

#### Model Version

```Python
class mlflow.entities.model_registry.ModelVersion(
    name, # String. Unique name within Model Registry.
    version,
    creation_timestamp: int, # Model version creation timestamp (milliseconds since the Unix epoch).
    last_updated_timestamp: int, # Timestamp of last update for this model version 

    description: str, # Description
    user_id: str, # User ID that created this model version.
    current_stage: str, #Current stage of this model version.
    source: str, # Source path for the model.
    run_id: str, # MLflow run ID that generated this model.
    status='READY', # Current Model Registry status for this model.
    status_message: str, # Descriptive message for error status conditions.
    tags: dict, # Dictionary of tag key (string) -> tag value for the current model version.
    run_link: str, # MLflow run link referring to the exact run that generated this model version.
    aliases: list[str] # List of aliases (string) for the current model version.
)
```

### Fetch model from registry

```Python
# Fetch a specific model version
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# Fetch a model version by alias
champion_version = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")

# Fetch the latest model version in a specific stage
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

# Search registered models or model versions with MlflowClient
client.search_registered_models(
    filter_string: Optional[str] = None,
    max_results: int = 100,
    order_by: Optional[List[str]] = None,
    page_token: Optional[str] = None) → PagedList[RegisteredModel]

client.search_model_versions(
    filter_string: Optional[str] = None,c
    max_results: int = 10000,
    order_by: Optional[List[str]] = None,
    page_token: Optional[str] = None) → PagedList[ModelVersion]

```

### Programmatically register a new model or new model version.

```Python
# Register model when logging
mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="sk-learn-random-forest-reg-model",
)

# Register already logged model
result = mlflow.register_model(
    "runs:/d16076a3ec534311817565e6527539c0/sklearn-model", "sk-learn-random-forest-reg"
)

# Create empty registered model and add version
client = MlflowClient()
client.create_registered_model("sk-learn-random-forest-reg-model")

result = client.create_model_version(
    name="sk-learn-random-forest-reg-model",
    source="mlruns/0/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
    run_id="d16076a3ec534311817565e6527539c0",
)
```


### Add metadata to a registered model and a registered model version

```Python
# Update description on model or model version
client.update_model_version(name: str, version: str, description: Optional[str] = None) → ModelVersion

client.update_registered_model(name: str, description: Optional[str] = None) → RegisteredModel

# Set or delete alias on registered model version
client.set_registered_model_alias(name: str, alias: str, version: str) 
client.delete_registered_model_alias(name: str, alias: str)

# Set or delete tag on a registered model
client.set_registered_model_tag(name, key, value)
client.delete_registered_model_tag(name: str, key: str)

```

### Identify, compare, and contrast the available model stages

```Python
# List available stages (['None', 'Staging', 'Production', 'Archived'])
client.get_model_version_stages(name: str, version: str) → List[str]

stage = model_version.current_stage

```

### Transition, archive, and delete model versions

```Python
from mlflow import MlflowClient

client = MlflowClient()

# Transition latest Staging model version to Production
staging_model_version = client.get_latest_versions(name, stages=["Staging"])[0]

client.transition_model_version_stage(
    name=model_name,
    version=staging_model_version.version,
    stage=['Production'],
    archive_existing_versions=True)

# Archive model version
client.transition_model_version_stage(
    name="sk-learn-random-forest-reg-model", version=3, stage="Archived")

# Delete model version
client.delete_model_version(name=model_name, version=model_version.version)
```

## Model Lifecycle Automation

### Identify the role of automated testing in ML CI/CD pipelines

### Describe how to automate the model lifecycle using Model Registry Webhooks and Databricks Jobs

### Identify advantages of using Job clusters over all-purpose clusters

### Describe how to create a Job that triggers when a model transitions between stages, given a scenario

### Describe how to connect a Webhook with a Job

### Identify which code block will trigger a shown webhook

### Identify a use case for HTTP webhooks and where the Webhook URL needs to come.

### Describe how to list all webhooks and how to delete a webhook