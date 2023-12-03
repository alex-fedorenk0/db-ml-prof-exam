# Section 2: Model Lifecycle Management

## Preprocessing Logic

### Describe an MLflow flavor and the benefits of using MLflow flavors

In MLflow, a "flavor" refers to a way of packaging and serving a model so that it can be easily consumed by different tools and environments. MLflow supports multiple flavors, and each flavor is designed to provide a standardized and consistent way to package, deploy, and serve machine learning models. The concept of flavors allows users to work with models across different frameworks and environments seamlessly. Some common flavors include Scikit-learn, TensorFlow, and PyTorch, each tailored to a specific framework.

Here's an overview of MLflow flavors and their benefits:

**MLflow Flavor**:
Definition: A flavor in MLflow is a convention for packaging models that allows users to easily switch between different machine learning frameworks without changing the model format.

**Benefits of Using MLflow Flavors**:
1. Framework Agnostic / Interoperability: MLflow flavors enable interoperability between different machine learning frameworks. You can train a model using one framework and serve it using another without the need for complex conversions.

2. Consistent Packaging / Standardized Format: MLflow provides a standardized format for packaging models, making it easy to share and deploy models across different platforms and environments.

3. Ease of Deployment / One Model, Many Environments: With MLflow flavors, a single model can be easily deployed in various environments, including cloud platforms, edge devices, and on-premises servers, regardless of the original training framework.
4. Simplified Model Serving / MLflow Model Servers: MLflow provides model serving capabilities that can load and serve models with different flavors. This makes it straightforward to deploy models in production without extensive modifications.
5. Versioning and Reproducibility / Model Versioning: Flavors help maintain version consistency across different models and frameworks. This is crucial for tracking changes, ensuring reproducibility, and managing the lifecycle of machine learning models.
6. Community Contributions / Extensible Framework: MLflow's extensible framework allows the community to contribute new flavors. This flexibility enables support for a wide range of machine learning frameworks and libraries.
7. Support for Custom Flavors / Customization: MLflow allows users to define custom flavors to handle specific model types or deployment scenarios. This enables users to extend MLflow's capabilities based on their unique requirements.
8. Easier Collaboration / Collaborative Development: Flavors simplify collaborative development by providing a common format for models. Team members can work on different stages of the ML lifecycle using their preferred frameworks, and the models can still be easily integrated and deployed.

In summary, MLflow flavors provide a powerful mechanism for packaging and serving machine learning models in a consistent and framework-agnostic manner. They contribute to the portability, reproducibility, and deployment ease of ML models across various environments and frameworks, facilitating collaboration and innovation in the machine learning community.

### Describe the advantages of using the pyfunc MLflow flavor

The python_function model flavor serves as a default model interface for MLflow Python models. Any MLflow Python model is expected to be loadable as a python_function model.

In addition, the mlflow.pyfunc module defines a generic filesystem format for Python models and provides utilities for saving to and loading from this format. The format is self contained in the sense that it includes all necessary information for anyone to load it and use it. Dependencies are either stored directly with the model or referenced via a Conda environment.

The mlflow.pyfunc module also defines utilities for creating custom pyfunc models using frameworks and inference logic that may not be natively included in MLflow. See Creating custom Pyfunc models.

### Describe the process and benefits of including preprocessing logic and context in custom model classes and objects

#### Including Preprocessing Logic and Context in Custom PyFunc Models

In MLflow, a custom PyFunc model is a way to deploy a machine learning model that requires custom logic or preprocessing steps before making predictions. A PyFunc model is a Python function that takes an input (e.g., features or raw data) and produces predictions. Including preprocessing logic and context in custom PyFunc model classes and objects allows you to encapsulate the entire pipeline, from data preprocessing to model inference, within a single deployable unit. This has several benefits:

##### Process of Including Preprocessing Logic and Context:

1. **Define a Custom PyFunc Class:**
   - Create a Python class that inherits from `mlflow.pyfunc.PythonModel`. This class will serve as the container for your model and preprocessing logic.

2. **Implement `load_context` and `predict` Methods:**
   - Implement the `load_context` method to load any additional context or configuration needed for preprocessing.
   - Implement the `predict` method, which takes the input data, performs preprocessing, and produces predictions using the trained model.

3. **Serialize and Save the Model:**
   - Serialize and save the entire PyFunc model, including the custom preprocessing logic and context, using MLflow tracking functions. This creates a deployable artifact.

4. **Deploy the Model:**
   - Deploy the saved PyFunc model as a REST API, a Docker container, or to a cloud service, depending on your deployment requirements.

5. **Make Predictions:**
   - Use the deployed PyFunc model to make predictions by sending input data to the deployed API or service.

##### Benefits of Including Preprocessing Logic and Context in Custom PyFunc Model Classes and Objects:

1. **Modularity:**
   - **Encapsulation:** All preprocessing logic is encapsulated within the PyFunc model, promoting modularity. This makes it easier to manage and update the preprocessing steps without affecting other components.

2. **Reproducibility:**
   - **Context Serialization:** By including the context or configuration needed for preprocessing in the PyFunc model, you ensure that the preprocessing steps are reproducible when deploying the model.

3. **Deployment Consistency:**
   - **Portable Deployment:** Including preprocessing logic in the PyFunc model ensures that the same preprocessing steps are applied consistently during both training and deployment, minimizing the risk of deployment issues.

4. **Simplified API:**
   - **Single Endpoint:** Deploying a PyFunc model with preprocessing logic as a single endpoint simplifies the API for making predictions. Clients only need to send raw input data, and the PyFunc model takes care of the preprocessing and prediction.

5. **Ease of Model Management:**
   - **Artifact Versioning:** MLflow automatically tracks and versions the PyFunc model, including its preprocessing logic. This makes it easy to manage and roll back to specific versions of both the model and preprocessing steps.

6. **Customization and Flexibility:**
   - **Custom Preprocessing Steps:** Including preprocessing logic in a custom PyFunc class allows you to define custom preprocessing steps tailored to your specific use case or domain.

7. **Debugging and Logging:**
   - **Traceability:** Preprocessing logic included in the PyFunc model can be logged and traced along with the model training process, facilitating debugging and auditing.

By including preprocessing logic and context in custom PyFunc model classes and objects, you create a self-contained and deployable unit that simplifies model management, promotes reproducibility, and ensures consistent preprocessing during both training and deployment.



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

Automated testing plays a crucial role in Machine Learning (ML) Continuous Integration/Continuous Deployment (CI/CD) pipelines. The primary goals of incorporating automated testing into ML CI/CD pipelines are to ensure the reliability, correctness, and robustness of ML models and their associated code. Here are some key roles and benefits of automated testing in ML CI/CD pipelines:

**Model Quality Assurance**:

- Regression Testing: Automated testing helps identify regressions by ensuring that changes in the codebase do not negatively impact the performance of existing models. This is particularly important as new features or code changes may unintentionally degrade the accuracy or other metrics of a model.

- Validation Testing: Automated tests can verify that the model's predictions match expected outcomes, ensuring that the model generalizes well to new data and is not overfitting or underfitting.

**Code Quality Assurance**:

- Unit Testing: Unit tests verify the correctness of individual components or functions within the ML codebase. This is crucial for detecting errors in data preprocessing, feature engineering, and model training steps.

- Integration Testing: Integration tests assess the interactions between different parts of the ML system, including data pipelines, model training, and inference components. This ensures that the integrated system behaves as expected.

**Data Pipeline Testing**:

- Data Quality Testing: Automated testing helps ensure the quality of input data by validating its format, distribution, and consistency. This is vital for maintaining the integrity of the training and testing datasets.

- Data Drift Testing: Tests can be designed to detect unexpected changes in the statistical properties of the data, helping to identify and mitigate issues related to data drift.

**Performance and Scalability Testing**:

- Scalability Testing: Automated tests can evaluate how well the ML system scales with an increasing volume of data, ensuring that the system can handle larger datasets without significant performance degradation.

- Latency and Throughput Testing: Performance tests assess the response time and throughput of the ML model during inference, helping to identify potential bottlenecks and optimize the system for efficiency.

**Deployment Validation**:

- Canary Testing: Before deploying a new model version to production, automated tests can be run on a subset of the production environment (canary deployment) to validate its performance and catch any issues before a full rollout.

- A/B Testing: Automated tests can be integrated into A/B testing frameworks to compare the performance of different model versions in a controlled and statistically sound manner.

**Continuous Feedback and Monitoring**:

- Monitoring Integration: Automated tests can be part of a broader monitoring strategy, providing continuous feedback on the health and performance of ML models in production.

- Alerting Systems: Integration with alerting systems ensures that any unexpected behavior or degradation in model performance triggers notifications for rapid response.

In summary, automated testing in ML CI/CD pipelines is essential for maintaining the quality and reliability of machine learning models throughout their lifecycle, from development and training to deployment and monitoring in production environments. It helps teams catch errors early, validate changes, and ensure that models continue to meet performance requirements over time

#### Steps for CI/CD on Databricks
A typical configuration for a Databricks CI/CD pipeline includes the following steps.

##### Continuous integration
Set up version control: Store your Databricks code and notebooks in a version control system like Git. This allows you to track changes over time and collaborate with other team members. See CI/CD techniques with Git and Databricks Repos.

**Code**: Develop code and unit tests in a Databricks notebook or using an external IDE.

**Build**: Automate the build process of your Databricks workspace using tools like Azure DevOps, Jenkins, or GitHub Actions. Through automation, you can build code consistently and integrate changes into your workspace.

**Test**: Develop and run automated tests to validate your code changes using tools like pytest or the Databricks CLI for automation.

**Release**: Generate a release package.

##### Continuous delivery
**Deploy**: Use a deployment tool like the Databricks CLI or the REST API to automate the deployment of your code changes to your Databricks workspace. You can also use the Azure DevOps release pipeline to deploy your code.

**Monitor**: Monitor the performance of your code and workflows in Databricks using tools like Azure Monitor or Datadog. This helps you identify and resolve any issues that arise in your production environment.

**Iterate**: Make small, frequent iterations to improve and update your data engineering or data science project. Small changes are easier to roll back than large ones.

### Describe how to automate the model lifecycle using Model Registry Webhooks and Databricks Jobs

Webhooks enable you to listen for Model Registry events so your integrations can automatically trigger actions. You can use webhooks to automate and integrate your machine learning pipeline with existing CI/CD tools and workflows. For example, you can trigger CI builds when a new model version is created or notify your team members through Slack each time a model transition to production is requested.

Webhooks are available through the Databricks REST API or the Python client `databricks-registry-webhooks` on PyPI.

#### Webhook events
You can specify a webhook to trigger upon one or more of these events:

- MODEL_VERSION_CREATED: A new model version was created for the associated model.
- MODEL_VERSION_TRANSITIONED_STAGE: A model version’s stage was changed.
- TRANSITION_REQUEST_CREATED: A user requested a model version’s stage be transitioned.
- COMMENT_CREATED: A user wrote a comment on a registered model.
- REGISTERED_MODEL_CREATED: A new registered model was created. This event type can only be specified for a registry-wide webhook, which can be created by not specifying a model name in the create request.
- MODEL_VERSION_TAG_SET: A user set a tag on the model version.
- MODEL_VERSION_TRANSITIONED_TO_STAGING: A model version was transitioned to staging.
- MODEL_VERSION_TRANSITIONED_TO_PRODUCTION: A model version was transitioned to production.
- MODEL_VERSION_TRANSITIONED_TO_ARCHIVED: A model version was archived.
- TRANSITION_REQUEST_TO_STAGING_CREATED: A user requested a model version be transitioned to staging.
- TRANSITION_REQUEST_TO_PRODUCTION_CREATED: A user requested a model version be transitioned to production.
- TRANSITION_REQUEST_TO_ARCHIVED_CREATED: A user requested a model version be archived.Types of webhooks

#### Types of webhooks based on their trigger targets

Webhooks with HTTP endpoints (HTTP registry webhooks): Send triggers to an HTTP endpoint.

Webhooks with job triggers (job registry webhooks): Trigger a job in a Databricks workspace. If IP allowlisting is enabled in the job’s workspace, you must allowlist the workspace IPs of the model registry. See IP allowlisting for job registry webhooks for more information.

There are also two types of webhooks based on their scope, with different access control requirements:

- **Model-specific webhooks**: The webhook applies to a specific registered model. You must have Can Manage permissions on the registered model to create, modify, delete, or test model-specific webhooks.

- **Registry-wide webhooks**: The webhook is triggered by events on any registered model in the workspace, including the creation of a new registered model. To create a registry-wide webhook, omit the model_name field on creation. You must have workspace admin permissions to create, modify, delete, or test registry-wide webhooks.


### Identify advantages of using Job clusters over all-purpose clusters

When you create a cluster you select a cluster type: an all-purpose cluster or a job cluster. All-purpose clusters can be shared by multiple users and are best for performing ad-hoc analysis, data exploration, or development. Once you’ve completed implementing your processing and are ready to operationalize your code, switch to running it on a job cluster. Job clusters terminate when your job ends, reducing resource usage and cost.

### Describe how to create a Job that triggers when a model transitions between stages, given a scenario

### Describe how to connect a Webhook with a Job

```bash
$ curl -X POST -H "Authorization: Bearer <access-token>" -d \ '{"model_name": "<model-name>",
  "events": ["TRANSITION_REQUEST_CREATED"],
  "description": "Job webhook trigger",
  "status": "TEST_MODE",
  "job_spec": {
    "job_id": "1",
    "workspace_url": "https://my-databricks-workspace.com",
    "access_token": "dapi12345..."}}'
https://<databricks-instance>/api/2.0/mlflow/registry-webhooks/create
```

```Python
from databricks_registry_webhooks import RegistryWebhooksClient, JobSpec

job_spec = JobSpec(
  job_id="1",
  workspace_url="https://my-databricks-workspace.com",
  access_token="dapi12345..."
)
job_webhook = RegistryWebhooksClient().create_webhook(
  model_name="<model-name>",
  events=["TRANSITION_REQUEST_CREATED"],
  job_spec=job_spec,
  description="Job webhook trigger",
  status="TEST_MODE"
)
```

### Identify which code block will trigger a shown webhook

### Identify a use case for HTTP webhooks and where the Webhook URL needs to come.

1. Create HTTP registry webhook
```bash
$ curl -X POST -H "Authorization: Bearer <access-token>" -d \
'{"model_name": "<model-name>",
  "events": ["MODEL_VERSION_CREATED"],
  "description": "Slack notifications",
  "status": "TEST_MODE",
  "http_url_spec": {
    "url": "https://hooks.slack.com/services/...",
    "secret": "anyRandomString"
    "authorization": "Bearer AbcdEfg1294"}}' https://<databricks-instance>/api/2.0/mlflow/registry-webhooks/create
```

```Python
from databricks_registry_webhooks import RegistryWebhooksClient, HttpUrlSpec

http_url_spec = HttpUrlSpec(
  url="https://hooks.slack.com/services/...",
  secret="secret_string",
  authorization="Bearer AbcdEfg1294"
)
http_webhook = RegistryWebhooksClient().create_webhook(
  model_name="<model-name>",
  events=["MODEL_VERSION_CREATED"],
  http_url_spec=http_url_spec,
  description="Slack notifications",
  status="TEST_MODE"
)
```
Response
```
{"webhook": {
   "id":"1234567890",
   "creation_timestamp":1571440826026,
   "last_updated_timestamp":1582768296651,
   "status":"TEST_MODE",
   "events":["MODEL_VERSION_CREATED"],
   "http_url_spec": {
     "url": "https://hooks.slack.com/services/...",
     "enable_ssl_verification": True
}}}
```

2. Test the webhook
```bash
$ curl -X POST -H "Authorization: Bearer <access-token>" -d \
'{"id": "1234567890"}' \
https://<databricks-instance>/api/2.0/mlflow/registry-webhooks/test
```
```Python
from databricks_registry_webhooks import RegistryWebhooksClient

http_webhook = RegistryWebhooksClient().test_webhook(
  id="1234567890"
)
```
Response:
```json
{
 "status":200,
 "body":"OK"
}
```

3. Update the webhook to active status

```bash
$ curl -X PATCH -H "Authorization: Bearer <access-token>" -d \
'{"id": "1234567890", "status": "ACTIVE"}' \
https://<databricks-instance>/api/2.0/mlflow/registry-webhooks/update
```

```Python
from databricks_registry_webhooks import RegistryWebhooksClient

http_webhook = RegistryWebhooksClient().update_webhook(
  id="1234567890",
  status="ACTIVE"
)
```

response
```json
{"webhook": {
   "id":"1234567890",
   "creation_timestamp":1571440826026,
   "last_updated_timestamp":1582768296651,
   "status": "ACTIVE",
   "events":["MODEL_VERSION_CREATED"],
   "http_url_spec": {
     "url": "https://hooks.slack.com/services/...",
     "enable_ssl_verification": True
}}}
```

### Describe how to list all webhooks and how to delete a webhook

#### List webhooks
```bash
$ curl -X GET -H "Authorization: Bearer <access-token>" -d \ '{"model_name": "<model-name>"}'
https://<databricks-instance>/api/2.0/mlflow/registry-webhooks/list
```

```Python
from databricks_registry_webhooks import RegistryWebhooksClient

webhooks_list = RegistryWebhooksClient().list_webhooks(model_name="<model-name>")
```
Response:
```json
{"webhooks": [{
   "id":"1234567890",
   "creation_timestamp":1571440826026,
   "last_updated_timestamp":1582768296651,
   "status": "ACTIVE",
   "events":["MODEL_VERSION_CREATED"],
   "http_url_spec": {
     "url": "https://hooks.slack.com/services/...",
     "enable_ssl_verification": True
}},
{
   "id":"1234567891",
   "creation_timestamp":1591440826026,
   "last_updated_timestamp":1591440826026,
   "status":"TEST_MODE",
   "events":["TRANSITION_REQUEST_CREATED"],
   "job_spec": {
     "job_id": "1",
     "workspace_url": "https://my-databricks-workspace.com"
}}]}

```

#### Delete a webhook

```bash
$ curl -X DELETE -H "Authorization: Bearer <access-token>" -d \
'{"id": "1234567890"}' \
https://<databricks-instance>/api/2.0/mlflow/registry-webhooks/delete
```

```Python
from databricks_registry_webhooks import RegistryWebhooksClient

http_webhook = RegistryWebhooksClient().delete_webhook(
  id="1234567890"
)
```