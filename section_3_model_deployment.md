# Section 3: Model Deployment

## Batch

### Describe batch deployment as the appropriate use case for the vast majority of deployment use cases

### Identify how batch deployment computes predictions and saves them somewhere for later use

### Identify live serving benefits of querying precomputed batch predictions

### Identify less performant data storage as a solution for other use cases

### Load registered models with load_model

### Deploy a single-node model in parallel using spark_udf

### Identify z-ordering as a solution for reducing the amount of time to read predictions from a table

### Identify partitioning on a common column to speed up querying

### Describe the practical benefits of using the score_batch operation


## Streaming

### Describe Structured Streaming as a common processing tool for ETL pipelines

### Identify structured streaming as a continuous inference solution on incoming data

### Describe why complex business logic must be handled in streaming deployments

### Identify that data can arrive out-of-order with structured streaming

### Identify continuous predictions in time-based prediction store as a scenario for streaming deployments

### Convert a batch deployment pipeline inference to a streaming deployment pipeline

### Convert a batch deployment pipeline writing to a streaming deployment pipeline


## Real-time

### Describe the benefits of using real-time inference for a small number of records or when fast prediction computations are needed

### Identify JIT feature values as a need for real-time deployment

### Describe model serving deploys and endpoint for every stage

### Identify how model serving uses one all-purpose cluster for a model deployment

### Query a Model Serving enabled model in the Production stage and Staging stage

### Identify how cloud-provided RESTful services in containers is the best solution for production-grade real-time deployments