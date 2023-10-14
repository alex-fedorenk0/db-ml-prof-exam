# Preparation for Databricks Certified Machine Learning Professional exam

## Exam outline

1. Experimentation - 30% (18 questions)
2. Model Lifecycle Management - 30% (18 questions)
3. Model Deployment - 25% (15 questions)
4. Solution and Data Monitoring - 15% (9 questions)

## Section 1. Experimentation

### Data Management
- Read and write a Delta table
- View Delta table history and load a previous version of a Delta table
- Create, overwrite, merge, and read Feature Store tables in machine learning
workflows

### Experiment Tracking
- Manually log parameters, models, and evaluation metrics using MLflow
- Programmatically access and use data, metadata, and models from MLflow
experiments

### Advanced Experiment Tracking
- Perform MLflow experiment tracking workflows using model signatures and input
examples
- Identify the requirements for tracking nested runs
- Describe the process of enabling autologging, including with the use of Hyperopt
- Log and view artifacts like SHAP plots, custom visualizations, feature data, images,
and metadata

## Section 2: Model Lifecycle Management

### Preprocessing Logic
- Describe an MLflow flavor and the benefits of using MLflow flavors
- Describe the advantages of using the pyfunc MLflow flavor
- Describe the process and benefits of including preprocessing logic and context in
custom model classes and objects

### Model Management
- Describe the basic purpose and user interactions with Model Registry
- Programmatically register a new model or new model version.
- Add metadata to a registered model and a registered model version
- Identify, compare, and contrast the available model stages
- Transition, archive, and delete model versions

### Model Lifecycle Automation
- Identify the role of automated testing in ML CI/CD pipelines
- Describe how to automate the model lifecycle using Model Registry Webhooks and
Databricks Jobs
- Identify advantages of using Job clusters over all-purpose clusters
- Describe how to create a Job that triggers when a model transitions between stages,
given a scenario
- Describe how to connect a Webhook with a Job
- Identify which code block will trigger a shown webhook
- Identify a use case for HTTP webhooks and where the Webhook URL needs to come.
- Describe how to list all webhooks and how to delete a webhook

## Section 3: Model Deployment

### Batch
- Describe batch deployment as the appropriate use case for the vast majority of
deployment use cases
- Identify how batch deployment computes predictions and saves them somewhere
for later use
- Identify live serving benefits of querying precomputed batch predictions
- Identify less performant data storage as a solution for other use cases
- Load registered models with load_model
- Deploy a single-node model in parallel using spark_udf
- Identify z-ordering as a solution for reducing the amount of time to read predictions
from a table
- Identify partitioning on a common column to speed up querying
- Describe the practical benefits of using the score_batch operation

### Streaming
- Describe Structured Streaming as a common processing tool for ETL pipelines
- Identify structured streaming as a continuous inference solution on incoming data
- Describe why complex business logic must be handled in streaming deployments
- Identify that data can arrive out-of-order with structured streaming
- Identify continuous predictions in time-based prediction store as a scenario for
streaming deployments
- Identify continuous predictions in time-based prediction store as a scenario for
streaming deployments
- Convert a batch deployment pipeline inference to a streaming deployment pipeline
- Convert a batch deployment pipeline writing to a streaming deployment pipeline

### Real-time
- Describe the benefits of using real-time inference for a small number of records or
when fast prediction computations are needed
- Identify JIT feature values as a need for real-time deployment
- Describe model serving deploys and endpoint for every stage
- Identify how model serving uses one all-purpose cluster for a model deployment
- Query a Model Serving enabled model in the Production stage and Staging stage
- Identify how cloud-provided RESTful services in containers is the best solution for
production-grade real-time deployments

## Section 4: Solution and Data Monitoring

### Drift Types
- Compare and contrast label drift and feature drift
- Identify scenarios in which feature drift and/or label drift are likely to occur
- Describe concept drift and its impact on model efficacy

### Drift Tests and Monitoring
- Describe summary statistic monitoring as a simple solution for numeric feature drift
- Describe mode, unique values, and missing values as simple solutions for categorical
feature drift
- Describe tests as more robust monitoring solutions for numeric feature drift than
simple summary statistics
- Describe tests as more robust monitoring solutions for categorical feature drift than
simple summary statistics
- Compare and contrast Jenson-Shannon divergence and Kolmogorov-Smirnov tests
for numerical drift detection
- Identify a scenario in which a chi-square test would be useful
### Comprehensive Drift Solutions
- Describe a common workflow for measuring concept drift and feature drift
- Identify when retraining and deploying an updated model is a probable solution to
drift
- Test whether the updated model performs better on the more recent data
