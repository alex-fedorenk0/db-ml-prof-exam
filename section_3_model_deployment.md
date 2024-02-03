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

Structured Streaming is a scalable and fault-tolerant stream processing engine built into the Apache Spark and the Spark SQL API. It allows you to express your streaming computation in the same way you would express a batch computation on static data. 

The Spark SQL engine will take care of running it incrementally and continuously, updating the final result as streaming data continues to arrive.

Structured Streaming has built-in support for a variety of data sources, such as Kafka, Flume, and Kinesis, allowing it to fit seamlessly into many existing ETL pipelines. Furthermore, the data can be manipulated using operations like filter, map and reduce.

As part of an ETL pipeline, Structured Streaming can extract data in real-time, transform it by cleaning, aggregating, or summarizing, and then load it into a system that can support subsequent queries or analysis, enabling real-time analytics and real-time applications.

Key features:

- Fast processing: Structured Streaming can process millions of records per second per node.
- Scalability: The platform is built on Spark, which is capable of scaling from a single machine to thousands of machines.
- Fault tolerance: Duplicates and missed data are addressed via the use of exactly-once semantics.
- Powerful window-based aggregations: It offers the ability to aggregate data over window intervals.
- Integration with Spark’s machine learning and graph processing algorithms: You can apply Spark’s machine learning libraries to data streams.

Overall, Structured Streaming is a robust tool for ETL processes that require real-time processing and analytics.

### Identify structured streaming as a continuous inference solution on incoming data

Structured Streaming is a scalable and fault-tolerance stream processing engine on the data platform. It allows for continuous data ingestion, processing and analytics in real-time.

In the context of machine learning, structured streaming can be used as a tool for continuous inference - making predictions as new data arrives. It facilitates real-time machine learning by continuously updating the ML model with the incoming data and by giving real-time predictions.

It allows users to express their streaming computation the same way as they would express a batch computation on static data, simplifying the machine learning process. Moreover, with its DataFrame and Dataset APIs, it promotes clean and easy-to-understand algorithms, enhancing the machine learning pipeline.

Overall, structured streaming is an ideal solution for applications that require real-time decision making such as fraud detection, live trend detection, and many more.

### Describe why complex business logic must be handled in streaming deployments
Handling complex business logic in streaming deployments is essential because real-time data provides unique opportunities to make immediate decisions that can have a significant impact on the business. Here's why:

1. **Real-Time Insights**: Streaming data allows businesses to apply complex logic on data as it is being produced, providing real-time insights and enabling faster decision making. 

2. **Predictive Analysis**: Streaming data also facilitates predictive analysis as models can be trained on real-time data, making predictions more accurate and up-to-date.

3. **Error Detection**: Complex business logic can be applied to detect anomalies or errors instantly, which is critical for cybersecurity, quality control, and system health monitoring. 

4. **Customer Engagement**: In scenarios like personalized recommendations or targeted advertisements, applying complex business logic on streaming data can greatly enhance customer engagement and satisfaction.

5. **Competitive Advantage**: Organizations that can implement complex business logic on streams of data are often in a better position to respond to changing market conditions, potentially gaining a competitive advantage.

Therefore, businesses must architect their streaming deployments to support complex logic to fully utilize the benefits of real-time data processing.

### Identify that data can arrive out-of-order with structured streaming

Absolutely, in structured streaming, data can indeed arrive out-of-order. This may happen due to reasons such as variations in the data production rate at the source, or network-related issues causing delays in data transmission. 

Structured Streaming allows the handling of such scenarios by providing capabilities like event-time processing and watermarks. 

- **Event-Time Processing** allows the system to handle events that may have occurred at a different point in time, but arrive late, or out of order.

- **Watermarks** are used to define a threshold delay, beyond which, the system assumes that it will not see any more old data. This allows the system to handle late data and limit the state size.

These features enable Structured Streaming to gracefully deal with out-of-order data and ensure appropriate processing based on event time, not just the time of arrival at the system.

### Identify continuous predictions in time-based prediction store as a scenario for streaming deployments
Yes, continuous predictions in a time-based prediction store is an essential scenario for streaming deployments.

This scenario often appears in recommendation systems, predictive maintenance, real-time anomaly detection, and other applications where model predictions need to be continuously updated as new data streams in. For example, in predictive maintenance for heavy machinery, a model might predict the probability of a machine component's failure based on real-time sensor data. The predictions are stored with a timestamp and can be fetched for analysis or monitored to trigger alerts when the probability crosses a certain threshold.

Structured streaming plays a pivotal role in these situations. It continually consumes data from the data source, applies the trained machine learning model on this data to make new predictions, and writes these predictions into a time-based prediction store, which could be a database optimized for time-series data.

Having this capability allows businesses to make proactive decisions based on up-to-date insights, driving further efficiency and performance.

### Convert a batch deployment pipeline inference to a streaming deployment 

Converting a batch deployment pipeline inference in Databricks to a streaming deployment pipeline involves several steps:

**1. Define Stream from Source:**
Instead of reading data from a static source like a file or a database, in a streaming deployment, you need to define a stream from a real-time data source. Databricks supports a variety of streaming sources including Kafka, Flume, and Kinesis among others.

```python
streamingDataFrame = spark.readStream.format('source').option('option', 'value').load()
```

**2. Apply Transformations:**
Similar to batch processing, you can apply transformations on a streaming DataFrame. Most transformations that are available for static DataFrame are supported for streaming DataFrames as well.

```python
transformedStream = streamingDataFrame.transform(transformationFunction)
```

**3. Define ML Model:**
Define your machine learning model. This could be a model you have previously trained on batch data. You can then progressively apply the inference model on the data as it is streaming in.

```python
predictions = model.transform(transformedStream)
```

**4. Write Stream to Sink:**
Once you have your transformed streaming DataFrame, you can write it out to a sink, which could be a file system, a database, or another streaming system. You can choose to trigger the streaming query manually or set it to trigger after an interval.

```python
query = predictions.writeStream\
        .outputMode('append')\
        .format('sink')\
        .option('option', 'value')\
        .start()
```

**5. Manage and Monitor Streaming Query:**
Starting a stream returns a StreamingQuery object that you can use to monitor and manage the stream. Databricks also provides a web-based user interface to monitor active streaming queries.

Remember that in the streaming context, it's important to consider factors such as handling late data and managing streaming offsets. The tools and strategies necessary for these factors are built into Structured Streaming.

### Convert a batch deployment pipeline writing to a streaming deployment pipeline
Converting a batch deployment pipeline that writes to a database or a file system to a streaming deployment pipeline in Databricks involves these few steps:

**1. Define Stream from Source:** Similar to the batch system, you need to read data from a source but this time the data comes in real-time.

    ```python
    streamingDataFrame = spark.readStream.format('source_format').option('option', 'value').load('source_path')
    ```

**2. Apply Transformations:** Apply necessary transformations to the data just as you would do in a batch pipeline. 

    ```python
    transformedStream = streamingDataFrame.transform(transformationFunction)
    ```

**3. Write the Streams:** Unlike in the batch system (where you use `write()` to write the DataFrame to a location), here you should use `writeStream` method to continuously write the streaming data to the defined sink.

    ```python
    query = transformedStream.writeStream\
        .outputMode('append')\
        .format('sink_format')\
        .option('option', 'value')\
        .start('sink_path')
    ```

   In this snippet, 'append' is the output mode indicating that only new records should be added to the output sink, which can be a directory in a file system, a database, or a dashboard.

**4. Monitor and Manage Streaming Query:** `writeStream.start()` returns a StreamingQuery object, which can be used to monitor the process. Databricks also provides a convenient web UI to visualize the streaming process.

```python
query.id    # get the unique identifier of the running query that persists across restarts from checkpoint data
query.runId # get the unique id of this run of the query, which will be generated at every start/restart
query.name  # get the name of the auto-generated or user-specified name
query.recentProgress # an array of the most recent progress updates for this query
```

Keep in mind that all transformations applied to the streaming DataFrames should be deterministic because they might be evaluated more than once.

## Real-time

### Describe the benefits of using real-time inference for a small number of records or when fast prediction computations are needed

Using real-time inference brings several benefits especially when dealing with a small number of records or when fast prediction computations are needed:

**1. Immediate Results:** Real-time inference provides immediate predictions, which is crucial in scenarios where rapid decision making is required such as in fraud detection, real-time bidding, or recommendation systems.

**2. Lower Latency:** Real-time inference features lower latency than batch processing, as each data point is processed individually upon arrival. This is extremely beneficial when rapid response times are needed.

**3. Real-Time Analytics:** With real-time inference, one can perform real-time analytics and gain insights from data as soon as it becomes available, maximizing the business value derived from the data.

**4. Enhance User Experience:** In use-cases like personalized recommendations, serving predictions in real time can significantly enhance the user experience and engagement.

**5. Scalability:** Real-time inference allows systems to be more scalable as compared to batch processing. This is because the processing load can be spread out over time, and resources can be elastically adjusted based on demand.

Hence, real-time inference is a highly beneficial approach for use-cases where dealing with a small number of records as it arrives or needing fast predictions.

### Identify JIT feature values as a need for real-time deployment

Just-in-Time (JIT) feature values indeed have significance in real-time deployment. They are especially needed in scenarios where the characteristics of the data may dynamically change over time, or the feature's value is only known or calculable at the time of prediction.

In a real-time deployment model, these feature values are calculated from real-time streaming data on-the-fly (Just-in-Time) instead of being pre-computed and stored. Examples of JIT feature values include user actions (click, viewed items, etc.) on a website, sensor readings in real-time systems, and transactional data in financial services.

These JIT feature values allow for more accurate and context-relevant predictions, improving the overall effectiveness of the deployed model in addressing dynamic business needs. Therefore, it is absolutely critical for the streaming deployment pipeline to accommodate such JIT feature computations.

To maintain JIT computations, one needs to ensure that there is an appropriate infrastructure for real-time data collection and processing, as well as systems to extract features from this data instantly, and a machine learning model capable of making predictions using these features.

### Describe model serving deploys and endpoint for every stage

### Identify how model serving uses one all-purpose cluster for a model deployment
Model serving in Databricks uses one all-purpose cluster per workspace for serving models. When Model Serving is enabled for the first time in a workspace, the platform automatically creates a new all-purpose cluster named "Shared Databricks Model Serving". 

This model-serving cluster is used to host all the models that are served in the workspace. The specifications of this cluster (like the number of nodes and the machine type) can be configured based on the prediction load and the expected response time.

Each model and each version of the model is containerized and run as separate tasks within this shared Databricks model serving cluster. This approach offers the following benefits:

**1. Resource Utilization:** Running all served models in a single all-purpose cluster allows for better resource utilization and management than creating separate clusters for each model.

**2. Cost-effective:** Since models share the resources of the all-purpose cluster, this alleviates the need for maintaining separate dedicated resources for each model, leading to cost savings.

**3. Simplicity:** Managing and monitoring a single cluster is notably simpler than handling multiple clusters.

**4. Scalability:** The all-purpose cluster can be easily scaled up or down to accommodate varying loads, ensuring that the model serving setup is both efficient and resilient.

However, do note that this cluster setup might not be suitable for all use cases and may depend on the model size, complexity, memory usage, and the prediction load. In some scenarios, you may still want to have dedicated resources (clusters) for individual models.

### Query a Model Serving enabled model in the Production stage and Staging stage

When you enable Model Serving in Databricks, it creates a REST API endpoint for the model. You can use this endpoint to interact with the model. Here is how you can query a Model Serving enabled model in the Production and Staging stages.

**1. Create an HTTP POST request:** Each model version and stage have their unique URL. By default, the base route for all model version URLs is /model. The URLs for the two versions would look like:

```
~/model/MyModel/Production/
~/model/MyModel/Staging/
```

**2. Send the POST request to the endpoint with your data:** Make sure your input data conforms with the format that your model expects – it could be a JSON, or a CSV string etc.

An example of a `curl` request to the production endpoint would be:

```curl
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["a", "b", "c"],"data":[[1, 2, 3],[4, 5, 6]]}' https://<your-databricks-url>/model/MyModel/Production
```

And for the staging endpoint:

```curl
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["a", "b", "c"],"data":[[1, 2, 3],[4, 5, 6]]}' https://<your-databricks-url>/model/MyModel/Staging
```

This will return the prediction from the model in the form of a JSON response. 

Remember to replace `<your-databricks-url>` with your actual Databricks domain. Make sure you have the necessary access rights to query the endpoint.

### Identify how cloud-provided RESTful services in containers is the best solution for production-grade real-time deployments

Using cloud-provided RESTful services in containers is indeed an effective solution for production-grade real-time deployments due to several reasons:

**1. Scalability:** Containers can be easily scaled up or down based on the demand. This ensures efficient resource utilization and can handle higher loads during peak times.

**2. Isolation:** Each container runs in an isolated environment which means it has its dedicated resources and environment variables. This prevents conflicts between different applications and ensures consistent behavior.

**3. Portability:** Containers encapsulate all the dependencies required to run an application. This makes the application portable and ensures it can run seamlessly across different development, staging, and production environments.

**4. Fast Startup Time:** Containers boot up much faster than virtual machines, providing faster response times for on-demand scalability.

**5. Deployment Flexibility:** By leveraging service meshes and orchestrators like Kubernetes, one can manage, automate, and scale deployments effortlessly.

**6. Version Control and Rollback Features:** Containers support version control which is beneficial in keeping track of the application changes. If anything goes wrong, with containers, you can easily revert to a previous stable state.

**7. Integration with DevOps:** Containers fit well into the DevOps methodology, as they allow for continuous integration and continuous deployment (CI/CD) of applications, leading to faster time-to-market and more iterative improvement.

**8. Cost-Efficient:** By maximizing resource utilization (running multiple containers on a single larger VM), cloud costs can be efficient.

Leveraging all these benefits, cloud providers wrap machine learning models in RESTful services, expose endpoints, and containerize them to support robust and production-grade real-time deployments seamlessly.
