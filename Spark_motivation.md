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
- reduce
- collect
- count
- first
- take
- saveAsTestFile
- countByKey

# Spark transformations -
- map
- filter
- flatMap
- mapPartitions
- mapPartitionsWithIndex
- sample
- union
- intersection
- distinct
- groupByKey
- reduceByKey
- aggregateByKey
- sortByKey
- join
- coalesce
- repartition

# Shuffle operations
Certain operations within Spark trigger an event known as the shuffle.
The shuffle is Spark’s mechanism for **re-distributing** data so that it’s grouped differently across partitions.
This typically involves copying data across executors and machines, making the shuffle **a complex and costly** operation.
Operations which can cause a shuffle include repartition operations like repartition and coalesce, ‘ByKey operations (except for counting)
like groupByKey and reduceByKey, and join operations like cogroup and join.
The Shuffle is an expensive operation since it involves disk I/O, data serialization, and network I/O

After each action spark forgets about the loaded data and any intermediate variables value you used in between.

So, if you invoke 4 actions one after another, it computes everything from beginning each time.

Reason is simple, spark works by building DAG, which lets it visualize path of operation from reading of data to action, and than it executes it.

# Caching

Each persisted RDD can be stored using a different storage level, allowing you, for example, to persist the dataset on disk, persist it in memory but as serialized Java objects (to save space), replicate it across nodes. These levels are set by passing a StorageLevel object (Scala, Java, Python) to persist().
The cache() method is a shorthand for using the default storage level, which is StorageLevel.MEMORY_ONLY (store deserialized objects in memory).
The available storage levels in Python include MEMORY_ONLY, MEMORY_ONLY_2, MEMORY_AND_DISK, MEMORY_AND_DISK_2, DISK_ONLY, and DISK_ONLY_2

- MEMORY_ONLY - Store RDD as deserialized Java objects in the JVM. If the RDD does not fit in memory,
some partitions will not be cached and will be recomputed on the fly each time they're needed. This is the default level.
- MEMORY_AND_DISK - Store RDD as deserialized Java objects in the JVM. If the RDD does not fit in memory, store the partitions that don't fit on disk,
and read them from there when they're needed.
- MEMORY_ONLY_SER(Java and Scala) - Store RDD as serialized Java objects (one byte array per partition). This is generally more space-efficient than deserialized objects,
especially when using a fast serializer, but more CPU-intensive to read.
- MEMORY_AND_DISK_SER(Java and Scala) - Similar to MEMORY_ONLY_SER, but spill partitions that don't fit in memory to disk instead of recomputing them on the fly each time they're needed.
- DISK_ONLY	Store the RDD partitions only on disk.
- MEMORY_ONLY_2, MEMORY_AND_DISK_2, etc. - Same as the levels above, but replicate each partition on two cluster nodes.
- OFF_HEAP (experimental) - Similar to MEMORY_ONLY_SER, but store the data in off-heap memory. This requires off-heap memory to be enabled.

## Comparison between two options in Spark:

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
- For medium / large data sets (10s of Gigs or 100s of Gigs) serialized caching would be helpful.  Because this will not consume too much memory. And garbage collecting gigs of memory can be taxing.

> Keep in mind that Spark will automatically evict RDD partitions from Workers in an LRU manner. The LRU eviction happens independently on each Worker and depends on the available memory in the Worker.
> Note that cache() is an alias for persist(StorageLevel.MEMORY_ONLY) which may not be ideal for datasets larger than available cluster memory.
Each RDD partition that is evicted out of memory will need to be rebuilt from source (ie. HDFS, Network, etc) which is expensive.
> A better solution would be to use persist(StorageLevel.MEMORY_AND_DISK_ONLY) which will spill the RDD partitions to the Worker's local disk if they're evicted from memory.
In this case, rebuilding a partition only requires pulling data from the Worker's local disk which is relatively fast.
> You also have the choice of persisting the data as a serialized byte array by appending _SER as follows:  MEMORY_SER and MEMORY_AND_DISK_SER. This can save space,
but incurs an extra serialization/deserialization penalty. And because we're storing data as a serialized byte arrays, less Java objects are created and therefore GC pressure is reduced.
>
# Important Notes:
- RDD - more than twice slower in Python than in Java/Scala. Bad performance in Python compared to Java/Scala
- Dataframe - similar performance in Python and in Java/Scala
- you want 2-4 partitions for each CPU in your cluster
- The textFile method also takes an optional second argument for controlling the number of partitions of the file. By default, Spark creates one partition for each block of the file (blocks being 128MB by default in HDFS),
but you can also ask for a higher number of partitions by passing a larger value. Note that you cannot have fewer partitions than blocks.
- By default, each transformed RDD may be recomputed each time you run an action on it. However, you may also persist an RDD in memory using the persist (or cache) method, in which case Spark will keep the
elements around on the cluster for much faster access the next time you query it. There is also support for persisting RDDs on disk, or replicated across multiple nodes.
- To estimate the memory consumption of a particular object, use SizeEstimator’s estimate method
- We highly recommend using Kryo if you want to cache data in serialized form, as it leads to much smaller sizes than Java serialization
- The first thing to try if GC is a problem is to use serialized caching

# Unit Testing
Spark is friendly to unit testing with any popular unit test framework. Simply create a SparkContext in your test with the master URL set to local,
run your operations, and then call SparkContext.stop() to tear it down. Make sure you stop the context within a finally block or the test framework’s tearDown method, as Spark does not support two contexts running concurrently in the same program.

# Data Locality
Spark prefers to schedule all tasks at the best locality level, but this is not always possible. In situations where there is no unprocessed data on any idle executor,
Spark switches to lower locality levels. There are two options:
 a) wait until a busy CPU frees up to start a task on data on the same server, or
 b) immediately start a new task in a farther away place that requires moving data there.

> What Spark typically does is wait a bit in the hopes that a busy CPU frees up. Once that timeout expires, it starts moving the data from far away to the free CPU.
The wait timeout for fallback between each level can be configured individually or all together in one parameter; see the spark.locality parameters.
You should increase these settings if your tasks are long and see poor locality, but the default usually works well.


# References
1. https://medium.com/@thejasbabu/spark-under-the-hood-partition-d386aaaa26b7
2. http://sujee.net/2015/01/22/understanding-spark-caching/#.WrDdf5PwYY0
3. Storage Levels - https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.storage.StorageLevel$