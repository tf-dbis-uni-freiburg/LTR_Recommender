from pyspark.ml.base import Transformer
import pyspark.sql.functions as F
from spark_utils import UDFContainer

class PapersPairBuilder(Transformer):
    """
    Class that builds pairs based on the representations of positive and negative papers. These representations are in form 
    vectors with the same size. It provides also different options on build the pairs. 
    For example, if a representation of a positive paper is p_p and a representation of its corresponding negative paper is n_p. 
    There are three options for pair building:
    1) duplicated_pairs - In the given example above, the difference between (p_p - p_n) will be computed and added with class 1.
    As well as (p_n - p_p) with class 0.
    2) one_class_pairs - only the difference between (p_p - p_n) will be computed and added with class 1.
    3) equally_distributed_pairs - because each positive paper p_p has a set of corresponding negative papers, for 50% of them (p_p - p_n) will 
    be computed and added wirh class 1. And for the other 50%, (p_n - p_p) with class 0.
    """

    def __init__(self, pairs_generation):
        """
        Constructs the builder.

        :param pairs_generation: there are three possible values: duplicated_pairs, one_class_pairs, 
        equally_distributed_pairs.For example, if we have a paper p_p, and a set of negative papers N for the paper p
        1) if it is "duplicated_pairs" - for each paper n_p of the set N, calculate (p_p - p_n, class:1) and 
        (p_n - p_p, class: 0)
        2) if it is "one_class_pairs" - for each paper n_p of the set N, calculate (p_p - p_n, class:1)
        3) if it is "equally_distributed_pairs" - for 50% of papers in the set N, calculate p_p - p_n, class:1), 
        and for the other 50% (p_n - p_p, class: 0)
        """
        self.pairs_generation = pairs_generation

    def _transform(self, dataset):
        if(self.pairs_generation == "equally_distributed_pairs"):
            # 50 % of the paper_pairs with label 1, 50% with label 0
            print("sy")
        elif(self.pairs_generation == "duplicated_pairs"):
            # add the difference (positive_paper_vector - negative_paper_vector) with label 1
            positive_class_dataset = dataset.withColumn("paper_pair_diff",
                                         UDFContainer.getInstance().vector_diff_udf("positive_paper_tf_vector", "negative_paper_tf_vector"))
            # add label 1
            positive_class_dataset = positive_class_dataset.withColumn("label", F.lit(1))

            # add the difference (negative_paper_vector - positive_paper_vector) with label 0
            negative_class_dataset = dataset.withColumn("paper_pair_diff",
                                                        UDFContainer.getInstance().vector_diff_udf("negative_paper_tf_vector",
                                                                        "positive_paper_tf_vector"))
            # add label 0
            negative_class_dataset = negative_class_dataset.withColumn("label", F.lit(0))

            dataset = positive_class_dataset.union(negative_class_dataset)
        elif(self.pairs_generation == "one_class_pairs"):
            # add the difference (positive_paper_vector - negative_paper_vector) with label 1
            dataset = dataset.withColumn("paper_pair_diff",
                                         UDFContainer.getInstance().vector_diff_udf("positive_paper_tf_vector", "negative_paper_tf_vector"))
            # add label 1
            dataset = dataset.withColumn("label", F.lit(1))
        else:
            # throw an error - unsupported option
            raise ValueError('The option' + self.pairs_generation + ' is not supported.')
        dataset = dataset.drop("positive_paper_tf_vector", "negative_paper_tf_vector")
        return dataset