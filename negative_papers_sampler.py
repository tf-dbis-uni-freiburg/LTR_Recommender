from pyspark.ml.base import Transformer
import pyspark.sql.functions as F
from random import randint
from pyspark.sql.types import *

"""
"""
class NegativePaperSampler(Transformer):

    paper_ids = []

    def __init__(self, spark, papers_corpus, k):
        """
        :param spark: spark instance used for broadcasting 
        :param papers_corpus: dataframe that contains all papers from the corpus
        :param k: number of negative papers that will be sampled per paper
        """
        self.spark = spark
        self.papers_corpus = papers_corpus
        self.k = k

    def _transform(self, dataset):

        # collect positive papers per user
        user_library = dataset.select("user_id", "paper_id").groupBy("user_id").agg(F.collect_list("paper_id").alias("positive_papers_per_user"))

        # all papers id in the paper corpus
        all_papers_id = self.papers_corpus.select("paper_id").distinct()

        # collect all paper ids in the corpus
        self.paper_ids = [int(i.paper_id) for i in all_papers_id.collect()]

        # broadcast the mapping
        broadcasted_paper_ids = self.spark.sparkContext.broadcast(self.paper_ids)

        total_papers_count = len(broadcasted_paper_ids.value)

        # generate k negative papers
        def generate_negatives(positives, total_papers_count, k):
            negatives = set()
            while len(negatives) < k:
                candidate = randint(1, total_papers_count + 1)
                # if a candidate paper is not in positives for an user and there exists paper with such id in the paper corpus
                if candidate not in positives and candidate in broadcasted_paper_ids.value and candidate not in negatives:
                    negatives.add(candidate)
            return list(negatives)

        generate_negatives_udf = F.udf(generate_negatives, ArrayType(IntegerType(), False))

        # add list of positive papers per user
        dataset = dataset.join(user_library, "user_id")
        # generate negative papers
        dataset = dataset.withColumn("negative_paper_ids", generate_negatives_udf("positive_papers_per_user", F.lit(total_papers_count), F.lit(self.k)))
        # drop columns that we have added
        dataset = dataset.drop("positive_papers_per_user")
        # explode negative paper ids column
        dataset = dataset.withColumn("negative_paper_id", F.explode("negative_paper_ids")).drop("negative_paper_ids")
        return dataset
