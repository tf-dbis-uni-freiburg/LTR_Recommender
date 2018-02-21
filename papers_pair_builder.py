from pyspark.ml.base import Transformer
import pyspark.sql.functions as F
from spark_utils import UDFContainer

class PapersPairBuilder(Transformer):
    """
    Class that builds pairs based on the representations of papers and their peers papers These representations are in form 
    vectors with the same size. It provides also different options on build the pairs. 
    For example, if a representation of a paper is p and a representation of its corresponding peer paper is p_p. 
    There are three options for pair building:
    1) duplicated_pairs - In the given example above, the difference between (p - p_p) will be computed and added with class 1.
    As well as (p_p - p) with class 0.
    2) one_class_pairs - only the difference between (p - p_p) will be computed and added with class 1.
    3) equally_distributed_pairs - because each paper p has a set of corresponding peer papers, for 50% of them (p - p_p) will 
    be computed and added wirh class 1. And for the other 50%, (p_p - p) with class 0.
    """

    def __init__(self, pairs_generation, paperId_col="paper_id", peer_paperId_col="peer_paper_id",
                 paper_vector_col="positive_paper_vector", peer_paper_vector_col="peer_paper_vector",
                 output_col="pair_paper_difference", label_col="label"):
        """
        Constructs the builder.


        :param pairs_generation: there are three possible values: duplicated_pairs, one_class_pairs, 
        equally_distributed_pairs.For example, if we have a paper p, and a set of peer papers N for the paper p
        1) if it is "duplicated_pairs" - for each paper p_p of the set N, calculate (p - p_p, class:1) and 
        (p_p - p, class: 0)
        2) if it is "one_class_pairs" - for each paper p_p of the set N, calculate (p - p_p, class:1)
        3) if it is "equally_distributed_pairs" - for 50% of papers in the set N, calculate p - p_p, class:1), 
        and for the other 50% (p_p - p, class: 0)
        :param paperId_col name of the column that contains paper id
        :param peer_paperId_col name of the column that contains peer paper id
        :param paper_vector_col name of the column that contains representation of the paper
        :param peer_paper_vector_col name of the column that contains representation of the peer paper
        :param output_col: name of the column where the result vector is stored
        :param label_col name of the column where the class of the pair/result is stored
        """
        self.pairs_generation = pairs_generation
        self.paperId_col = paperId_col
        self.peer_paperId_col = peer_paperId_col
        self.paper_vector_col = paper_vector_col
        self.peer_paper_vector_col = peer_paper_vector_col
        self.output_col = output_col
        self.label_col = label_col


    def _transform(self, dataset):
        if(self.pairs_generation == "equally_distributed_pairs"):
            # 50 % of the paper_pairs with label 1, 50% with label 0

            # get a list of peer paper ids per paper
            peers_per_paper = dataset.groupBy(self.paperId_col).agg(F.collect_list(self.netagive_paperId_col).alias("peers_per_paper"))

            # generate 50/50 distribution to positive/negative class
            peers_per_paper = peers_per_paper.withColumn("equally_distributed_papers", UDFContainer.getInstance().split_papers_udf("peers_per_paper"))

            # positive label 1
            positive_class_per_paper = peers_per_paper.withColumn("positive_class_papers", F.col("equally_distributed_papers")[0])

            positive_class_per_paper = positive_class_per_paper.select(self.paperId_col, F.explode("positive_class_papers").alias(self.peers_paperId_col))
            # fix this
            positive_class_dataset = dataset.join(positive_class_per_paper, (positive_class_per_paper[self.peer_paperId_col] == dataset[self.peer_paperId_col])
                                                                        & (positive_class_per_paper[self.paperId_col] == dataset[self.paperId_col]) )

            # add the difference (paper_vector - peer_paper_vector) with label 1
            positive_class_dataset = positive_class_dataset.withColumn(self.output_col, UDFContainer.getInstance().vector_diff_udf(
                                                            self.paper_vector_col, self.peer_paper_vector_col))
            # add label 1
            positive_class_dataset = positive_class_dataset.withColumn(self.label_col, F.lit(1))

            # negative label 0
            negative_class_per_paper = peers_per_paper.withColumn("negative_class_papers", F.col("equally_distributed_papers")[1])
            negative_class_per_paper = negative_class_per_paper.select(self.paperId_col, F.explode("negative_class_papers").alias(self.peer_paperId_col))
            negative_class_dataset = dataset.join(negative_class_per_paper, (negative_class_per_paper[self.peer_paperId_col] == dataset[self.peer_paperId_col]) \
                                                    & (negative_class_per_paper[self.paperId_col] == dataset[self.paperId_col]))

            # add the difference (peer_paper_vector - paper_vector) with label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.output_col, UDFContainer.getInstance().vector_diff_udf(self.peer_paper_vector_col,
                                                            self.paper_vector_col))
            # add label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.label_col, F.lit(0))
            dataset = positive_class_dataset.union(negative_class_dataset)
        elif(self.pairs_generation == "duplicated_pairs"):
            # add the difference (paper_vector - peer_paper_vector) with label 1
            positive_class_dataset = dataset.withColumn(self.output_col,
                                         UDFContainer.getInstance().vector_diff_udf(self.paper_vector_col, self.peer_paper_vector_col))
            # add label 1
            positive_class_dataset = positive_class_dataset.withColumn(self.label_col, F.lit(1))

            # add the difference (peer_paper_vector - paper_vector) with label 0
            negative_class_dataset = dataset.withColumn(self.output_col,
                                                        UDFContainer.getInstance().vector_diff_udf(self.peer_paper_vector_col,
                                                                        self.paper_vector_col))
            # add label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.label_col, F.lit(0))

            dataset = positive_class_dataset.union(negative_class_dataset)
        elif(self.pairs_generation == "one_class_pairs"):
            # add the difference (paper_vector - peer_paper_vector) with label 1
            dataset = dataset.withColumn(self.output_col,
                                         UDFContainer.getInstance().vector_diff_udf(self.paper_vector_col, self.peer_paper_vector_col))
            # add label 1
            dataset = dataset.withColumn(self.label_col, F.lit(1))
        else:
            # throw an error - unsupported option
            raise ValueError('The option' + self.pairs_generation + ' is not supported.')
        return dataset