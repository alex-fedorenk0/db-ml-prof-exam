# Section 3: Model Deployment

## Batch

### Describe batch deployment as the appropriate use case for the vast majority of deployment use cases

### Identify how batch deployment computes predictions and saves them somewhere for later use

### Identify live serving benefits of querying precomputed batch predictions

Live serving is a technique that allows you to query precomputed batch predictions from a data store, such as Delta Lake, instead of invoking the model on demand. This can have several benefits, such as:

Reducing latency: You can access the predictions faster by reading them from the data store, rather than waiting for the model to process the input data.
Improving scalability: You can handle a large volume of queries by distributing the load across the data store nodes, rather than overloading the model server.
Simplifying deployment: You can avoid the complexity of setting up and maintaining a model server, and use the existing data store infrastructure and APIs.
Enabling offline analysis: You can perform historical or batch analysis on the precomputed predictions, such as aggregating, filtering, or joining them with other data sources.
Using batch model deployment on Databricks, you can leverage MLflow to generate and store batch predictions in Delta Lake, and use Delta Live Tables or Spark Structured Streaming to query them in real time. For more information, see Deploy models for batch inference and prediction.

### Identify less performant data storage as a solution for other use cases

For less performant data storage in batch model deployment on Databricks, you might consider using a distributed file system like Apache Hadoop Distributed File System (HDFS) or a cloud-based storage solution such as Amazon S3 or Azure Blob Storage. These are suitable for storing large volumes of data but may not provide the low-latency access required for live serving. Utilizing these storage options can be effective for offline batch processing and model training where real-time responsiveness is not critical.

### Load registered models with load_model

### Deploy a single-node model in parallel using spark_udf

### Identify z-ordering as a solution for reducing the amount of time to read predictions from a table

**Z-Ordering for Improved Prediction Retrieval**

[Z-ordering](https://en.wikipedia.org/wiki/Z-order_curve) is a technique employed in data storage to enhance query performance, particularly in scenarios where efficient retrieval of subsets of data is crucial. This method involves organizing data based on a specific order of columns, creating a multi-dimensional index. When considering the task of reading predictions from a table, Z-ordering can be beneficial in the following ways:

1. **Column Organization:** Z-ordering arranges data, grouping similar values together across multiple columns. This clustering minimizes the amount of data that needs to be read for a given query.

2. **Improved Data Locality:** By organizing data with Z-ordering, queries focusing on specific columns or value ranges can benefit from improved data locality. This means that related data is stored closer together, reducing the need to access scattered parts of the storage.

3. **Efficient Queries:** Reading predictions from a table can be more efficient when the data is organized using Z-ordering. Queries that filter or aggregate based on certain columns are particularly optimized, enhancing overall query performance.

4. **Reduced I/O Overhead:** Z-ordering minimizes the amount of Input/Output (I/O) operations needed to retrieve relevant data. This reduction in I/O overhead contributes to faster query execution.

It's essential to consider the specific use case and query patterns to determine whether Z-ordering is a suitable optimization for prediction retrieval tasks, as its effectiveness depends on the nature of your data and access patterns.

### Identify partitioning on a common column to speed up querying

**Partitioning for Improved Query Performance**

Partitioning involves organizing data based on a common column, and it can significantly enhance query speed and overall database performance. When considering the task of querying, particularly in large datasets, partitioning on a common column offers several advantages:

1. **Data Organization:** Partitioning divides the dataset into smaller, more manageable subsets based on a common column, such as date, category, or location. This organization streamlines data access and retrieval.

2. **Reduced Scanning:** When querying on the partitioned column, the database system can skip or focus on specific partitions, reducing the amount of data that needs to be scanned. This minimizes the computational load and speeds up query execution.

3. **Parallel Processing:** Partitioning allows for parallel processing of queries across different partitions. This parallelism can significantly improve overall query performance, especially in distributed computing environments.

4. **Optimized Aggregation:** For tasks involving aggregation or filtering based on the common column, partitioning ensures that relevant data is grouped together, reducing the need to scan the entire dataset.

5. **Efficient Joins:** When joining tables on the partitioning column, the process is more efficient as the data is already organized in a way that aligns with the join conditions.

By carefully selecting the appropriate partitioning column based on the query patterns and workload, you can optimize database performance and accelerate querying operations, particularly in scenarios with large and diverse datasets.

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