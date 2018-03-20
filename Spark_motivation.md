# Why Spark?

# The jargon of Apache Spark

**Job**: A piece of code which reads some input from HDFS or local, performs some computation on the data and writes some output data.
**Stages**: Jobs are divided into stages. Stages are classified as a Map or reduce stages. A Stage is a combination of
transformations which does not cause any shuffling, pipelining as many narrow transformations (eg: map, filter etc) as possible.
**Tasks**: Each stage has some tasks, one task per partition. One task is executed on one partition of data on one executor (machine).
**DAG**: DAG stands for Directed Acyclic Graph, in the present context its a DAG of operators.
**Executor**: The process responsible for executing a task.
**Driver**: The program/process responsible for running the Job over the Spark Engine
**Master**: The machine on which the Driver program runs
**Slave**: The machine on which the Executor program runs

# How Spark Works?

1. When a user runs an action (like collect), the Graph is submitted to a DAG Scheduler.
The DAG scheduler divides operator graph into (map and reduce) stages.
2. A stage is comprised of tasks based on partitions of the input data.
The DAG scheduler pipelines operators together to optimize the graph. For e.g. Many map operators can be scheduled in a single stage.
The final result of a DAG scheduler is a set of stages.
3. The stages are passed on to the **Task Scheduler**. The task scheduler launches tasks via cluster manager.
(Spark Standalone/Yarn/Mesos). The task scheduler doesn’t know about dependencies among stages.
4. The Worker/Executor executes the tasks. A new JVM is started per job. The worker knows only about the code that is passed to it.

RDD are stored in partitions. When performing computations on RDDs, these partitions can be operated on in parallel
When you submit a Spark job, the driver implicitly converts the code containing transformations and actions performed on the RDDs into a logical Directed Acyclic Graph (DAG). The driver program also performs certain optimizations like pipelining transformations and then it
converts the logical DAG into physical execution plan with set of stages
When we apply a transformation on a RDD, the transformation is applied to each of its partition. Spark spawns a single Task for a single partition, which will run inside the executor JVM. Each stage contains as many tasks as partitions of the RDD and will perform the transformations (map, filter etc) pipelined in the stage.
When the data is key-value oriented, partitioning becomes imperative because for subsequent transformations on the RDD, there’s a fair amount of shuffling of data across the network. If similar keys or range of keys are stored in the same partition then the shuffling is minimized and the processing becomes substantially fast.
# Having too less partitions results in

- Less concurrency,
- Increase memory pressure for transformation which involves shuffle,
- More susceptible for data skew.
# However, having too many partitions might also have negative impact

Too much time spent in scheduling multiple tasks
# Spark actions
# Spark transformations


After each action spark forgets about the loaded data and any intermediate variables value you used in between.

So, if you invoke 4 actions one after another, it computes everything from beginning each time.

Reason is simple, spark works by building DAG, which lets it visualize path of operation from reading of data to action, and than it executes it.

# Caching
There exist two options in Spark:

1. Raw storage
- Pretty fast to process
- Can take up 2x-4x more space. For example, 100MB data cached could consume 350MB memory
- Can put pressure in JVM and JVM garbage collection
- Usage:
```
rdd.persist(StorageLevel.MEMORY_ONLY)
#or
rdd.cache()
```
2. Serialized
- Slower processing than raw caching. For example, 500MB data cached, a count() operation
takes ~27, 063 ms before caching, 1,802 ms using serialized caching and 130 ms with raw caching.
- Overhead is minimal. Serialized caching consumes almost the same amount of memory as RDD. For example, 500MB data could consume 537.6 MB memory
- less pressure on JVM and JVM garbage collection

- Usage:
```
rdd.persist(StorageLevel.MEMORY_ONLY_SER)
```

3. Conclusion about caching
- For small data sets (few hundred megs) we can use raw caching.  Even though this will consume more memory, the small size won’t put too much pressure on Java garbage collection.
- Raw caching is also good for iterative work loads
- For medium / large data sets (10s of Gigs or 100s of Gigs) serialized caching would be helpful.  Because this will not consume too much memory.  And garbage collecting gigs of memory can be taxing.

# References
1. https://medium.com/@thejasbabu/spark-under-the-hood-partition-d386aaaa26b7
2. http://sujee.net/2015/01/22/understanding-spark-caching/#.WrDdf5PwYY0