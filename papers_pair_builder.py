from pyspark.ml.base import Transformer
import pyspark.sql.functions as F
from pyspark.ml.linalg import Vectors, VectorUDT
from collections import defaultdict

class PapersPairBuilder(Transformer):

    """
    Three ways of generating papar pair classes. For example, we have a paper p_p, and a set of negative papers N
                    for the paper p
    1) for each paper n_p of the set N, calculate (p_p - p_n, class:1) and (p_n - p_p, class: -1)
    2) for each paper n_p of the set N, calculate (p_p - p_n, class:1)
    3) for 50% of papers in the set N, calculate p_p - p_n, class:1), and for the other 50% (p_n - p_p, class: -1)
    """
    def __init__(self, label_generation):
        # TODO make it with labels not with numbers
        self.label_generation = label_generation

    def _transform(self, dataset):

        def diff(v1, v2):
            values = defaultdict(float)  # Dictionary with default value 0.0
            # Add values from v1
            for i in range(v1.indices.size):
                values[v1.indices[i]] += v1.values[i]
            # subtract values from v2
            for i in range(v2.indices.size):
                values[v2.indices[i]] -= v2.values[i]
            return Vectors.sparse(v1.size, dict(values))

        # register the udf
        vector_diff_udf = F.udf(diff, VectorUDT())

        if(self.label_generation == 0):
            #sthhh
            # 50 % of the paper_pairs with label 1, 50% with label 0
            print("Has to be implemented")
        elif(self.label_generation == 1):
            # add the difference (positive_paper_vector - negative_paper_vector) with label 1
            positive_class_dataset = dataset.withColumn("paper_pair_diff",
                                         vector_diff_udf("positive_paper_tf_vector", "negative_paper_tf_vector"))
            # add label 1
            positive_class_dataset = positive_class_dataset.withColumn("label", F.lit(1))

            # add the difference (negative_paper_vector - positive_paper_vector) with label 0
            negative_class_dataset = dataset.withColumn("paper_pair_diff",
                                                        vector_diff_udf("negative_paper_tf_vector",
                                                                        "positive_paper_tf_vector"))
            # add label -1
            negative_class_dataset = negative_class_dataset.withColumn("label", F.lit(0))

            dataset = positive_class_dataset.union(negative_class_dataset)
        elif(self.label_generation == 2):
            # add the difference (positive_paper_vector - negative_paper_vector) with label 1
            dataset = dataset.withColumn("paper_pair_diff",
                                         vector_diff_udf("positive_paper_tf_vector", "negative_paper_tf_vector"))
            # add label 1
            dataset = dataset.withColumn("label", F.lit(1))
        else:
            # TODO throw an error - unsupported option
            print("Has to be implemented")

        dataset = dataset.drop("positive_paper_tf_vector", "negative_paper_tf_vector")
        return dataset