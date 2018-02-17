from pyspark.ml.base import Transformer
import pyspark.sql.functions as F
from spark_utils import *
from pyspark.sql.window import Window

class NegativePaperSampler(Transformer):
    """
    Generate negative papers for a paper. For each paper in the data set, a list of negative papers will be selected from the paper corpus.
    The negative papers are selected randomly.
    """

    def __init__(self, papers_corpus, k, paperId_col="paper_id", userId_col="user_id", output_col="negative_paper_id"):
        """
        :param papers_corpus: PaperCorpus object that contains data frame. It represents all papers from the corpus. 
        Possible format (paper_id, citeulike_paper_id). See PaperCorpus documentation for more information.
        :param k: number of negative papers that will be sampled per paper
        :param paperId_col name of the paper id column in the input data frame of transform()
        :param userId_coln name of the user id column in the input data frame of transform()
        :param output_col the name of the column in which the produced result is stored
        """
        self.papers_corpus = papers_corpus
        self.k = k
        self.paperId_col = paperId_col
        self.userId_col = userId_col
        self.output_col = output_col

    def _transform(self, dataset):
        """
        The input data set consists of (paper, user) pairs. Each paper is represented by paper id, each user by user id.
        The names of the columns that store them are @paperId_col and @userId_col, respectively.
        The method generates each pair, a list of negative papers. Each user has a library which is a list of paper ids 
        that the user likes. Negative papers for each paper are generated randomly from all papers in the paper corpus 
        except the papers that the user in the pair(paper,user) likes. At the end, output data set will have an additional 
        column "negative_paper_id" that contains one of the generated ids. The number of row per pair will be equal to the 
        number of generated negative papers. For example, if a paper corpus contains paper ids - [1, 2, 3, 4, 5, 6],
        a (user, paper) pair is (1, 2). The user library (user, [list of liked papers]) is (1, [1, 2, 3]). If the number 
        of generated papers for a pair is 2, for the example, there is three possibilities (4, 5), (5, 6) or (4, 6).
        
        :param dataset: mandatory columns @paperId_col and @negative_paper_id
        :return: data set with additional column @output_col
        """

        # because paper ids in the papers_corpus are not sequential, generate a column "paper_id_index" with sequential order
        papers = self.papers_corpus.papers.drop(self.papers_corpus.citeulikePaperId_col)
        indexed_papers_corpus = papers.withColumn('paper_id_index',
                                                               F.row_number().over(Window.orderBy(self.papers_corpus.paperId_col)))

        # add the generated index to each paper in the input dataset
        indexed_dataset = dataset.join(indexed_papers_corpus, indexed_papers_corpus[self.papers_corpus.paperId_col] == dataset[self.paperId_col]).drop(self.papers_corpus.paperId_col)

        total_papers_count = indexed_papers_corpus.count()

        # collect positive papers per user
        user_library = indexed_dataset.select(self.userId_col, "paper_id_index").groupBy(self.userId_col)\
                                        .agg(F.collect_list("paper_id_index").alias("indexed_positive_papers_per_user"))

        # add list of positive papers per user
        dataset = dataset.join(user_library, self.userId_col)

        # generate negative papers
        dataset = dataset.withColumn("indexed_negative_papers_per_user", UDFContainer.getInstance()
                                        .generate_negatives_udf("indexed_positive_papers_per_user", F.lit(total_papers_count), F.lit(self.k)))

        # drop columns that we have added
        dataset = dataset.drop("indexed_positive_papers_per_user")

        # explode negative paper ids column
        dataset = dataset.withColumn("indexed_negative_paper_id", F.explode("indexed_negative_papers_per_user"))\
            .drop("indexed_negative_papers_per_user")

        # rename the column before the join to not mix them
        dataset = dataset.withColumnRenamed(self.paperId_col, "positive_paper_id")

        # revert indexed_negative_paper_id to oridinal paper_id
        dataset = dataset.join(indexed_papers_corpus, dataset.indexed_negative_paper_id == indexed_papers_corpus.paper_id_index)
        dataset = dataset.drop("paper_id_index", "indexed_negative_paper_id")
        dataset = dataset.withColumnRenamed(self.paperId_col, self.output_col)
        dataset = dataset.withColumnRenamed("positive_paper_id", self.paperId_col)
        return dataset