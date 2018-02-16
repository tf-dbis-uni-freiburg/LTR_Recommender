from pyspark.ml.base import Transformer
import pyspark.sql.functions as F
from spark_utils import *
from pyspark.sql.window import Window

class NegativePaperSampler(Transformer):
    """
    Generate negative papers for a paper. For each paper in the data set, a list of negative papers will be selected from the paper corpus.
    The negative papers are selected randomly.
    """

    def __init__(self, papers_corpus, k):
        """
        :param papers_corpus: data frame that contains all papers from the corpus. Format (paper_id, citeulike_paper_id).
        :param k: number of negative papers that will be sampled per paper
        """
        self.papers_corpus = papers_corpus
        self.k = k

    def _transform(self, dataset):
        """
        The input data set consists of (paper, user) pairs. Each paper is represented by paper_id, each user by user_id.
        The method generates each pair, a list of negative papers. Each user has a library which is a list of paper ids that the user likes.
        Negative papers for each paper are generated randomly from all papers in the paper corpus except the papers that the user in the pair(paper,user) likes.
        At the end, output data set will have an additional column "negative_paper_id" that contains one of the generated ids. The number of row per pair will be
        equal to the number of generated negative papers.
        For example, if a paper corpus contains paper ids - [1, 2, 3, 4, 5, 6], a (user, paper) pair is (1, 2). The user library (user, [list of liked papers]) is
        (1, [1, 2, 3]). If the number of generated papers for a pair is 2, for the example, there is three possibilities (4, 5), (5, 6) or (4, 6).
        
        :param dataset: mandotory columns "paper_id" and "user_id"
        :return: dataset with additional column "negative_paper_id"
        """

        # because paper ids in the papers_corpus are not sequential, generate a column "paper_id_index" with sequential order
        self.papers_corpus = self.papers_corpus.drop("citeulike_paper_id")
        indexed_papers_corpus = self.papers_corpus.withColumn('paper_id_index',
                                                               F.row_number().over(Window.orderBy("paper_id")))

        # add the generated index to each paper in the input dataset
        indexed_dataset = dataset.join(indexed_papers_corpus, "paper_id")

        total_papers_count = indexed_papers_corpus.count()
        # collect positive papers per user
        user_library = indexed_dataset.select("user_id", "paper_id_index").groupBy("user_id").agg(F.collect_list("paper_id_index").alias("indexed_positive_papers_per_user"))

        # add list of positive papers per user
        dataset = dataset.join(user_library, "user_id")

        # generate negative papers
        dataset = dataset.withColumn("indexed_negative_papers_per_user", UDFContainer.getInstance().generate_negatives_udf("indexed_positive_papers_per_user", F.lit(total_papers_count), F.lit(self.k)))

        # drop columns that we have added
        dataset = dataset.drop("indexed_positive_papers_per_user")

        # explode negative paper ids column
        dataset = dataset.withColumn("indexed_negative_paper_id", F.explode("indexed_negative_papers_per_user")).drop("indexed_negative_papers_per_user")

        # rename the column before the join to not mix them
        dataset = dataset.withColumnRenamed("paper_id", "positive_paper_id")

        # revert indexed_negative_paper_id to oridinal paper_id
        dataset = dataset.join(indexed_papers_corpus, dataset.indexed_negative_paper_id == indexed_papers_corpus.paper_id_index)
        dataset = dataset.drop("paper_id_index", "indexed_negative_paper_id")
        dataset = dataset.withColumnRenamed("paper_id", "negative_paper_id")
        dataset = dataset.withColumnRenamed("positive_paper_id", "paper_id")
        return dataset