from pyspark.ml.base import Estimator
from pyspark.ml.base import Transformer
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from spark_utils import UDFContainer

class TFVectorizer(Estimator):
    """
    Calculates a profile for each paper. Its representation is based on tf scores.
    It is initialized with paper corpus - all papers taken into account when a representation
    of a paper is built.
    """

    def __init__(self, papers_corpus, paperId_col="paper_id", tf_map_col="term_occurrence", output_tf_col="tf_vector"):
        """
        Create an instance of the class.
        
        :param papers_corpus: PaperCorpus object that contains data frame. It represents all papers from the corpus. 
        Possible format (paper_id, citeulike_paper_id). See PaperCorpus documentation for more information.
        :param paperId_col name of the paper id column in the input data frame of fit()
        :param tf_map_col name of the tf representation column in the input data frame of fit(). The type of the 
        column is Map. It contains key:value pairs where key is the term id and value is #occurence of the term
        in a particular paper.
        :param output_tf_col the name of the column in which the produced result is stored - tf vector of a paper
        """
        self.papers_corpus = papers_corpus
        # data frame for a mapping between term_id and sequantial id based on all terms in the corpus
        self.term_mapping = None
        # data frame contains all the terms in the corpus and their corresponding ids
        self.term_corpus = None
        self.paperId_col= paperId_col
        self.tf_map_col = tf_map_col
        self.output_tf_col = output_tf_col

    def setPaperIdCol(self, paperId_col):
        self.paperId_col = paperId_col

    def setTFMapCol(self, tf_map_col):
        self.tf_map_col = tf_map_col

    def setOutputTFCol(self, output_tf_col):
        self.output_tf_col

    def _fit(self, papers):
        """
        Build a tf representation for each paper in the input data set. Based on papers in the papers corpus, a set of all
        terms is extracted. For each of them a unique id is generated. Term ids are sequential. Then depending on all terms
        and their frequence for a paper, a sparse vector is built. A model that can be used to map a tf vector to each paper 
        based on its paper id is returned.
    
        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: a build model which can be used for transformation of a data set
        """

        # select only those papers that are part of the paper corpus
        papers = papers.join(self.papers_corpus.papers, papers[self.paperId_col] == self.papers_corpus.papers[self.papers_corpus.paperId_col]).drop(papers[self.paperId_col])

        # explode map with key:value pairs
        # term_occurence
        exploded_papers = papers.select(self.paperId_col, F.explode(self.tf_map_col))

        # collect all distinct term ids
        terms = exploded_papers.select("key").distinct()

        # generate sequential ids for terms
        term_corpus = terms.withColumn('id', F.row_number().over(Window.orderBy("key")))
        self.term_corpus = term_corpus

        # join term corpus with exploded papers to add new id for each term
        # format (key, paper_id, value, id)
        indexed_exploded_papers = exploded_papers.join(term_corpus, "key")
        # collect (id, value) pairs into one list
        indexed_exploded_papers = indexed_exploded_papers.groupby(self.paperId_col).agg(F.collect_list(F.struct("id", "value")).alias("term_occurrence"))
        voc_size = terms.count()

        dataset = indexed_exploded_papers.withColumn(self.output_tf_col, UDFContainer.getInstance().to_tf_vector_udf("term_occurrence", F.lit(voc_size)))

        paper_profiles = dataset.select(self.paperId_col, self.output_tf_col)
        return TFVectorizorModel(paper_profiles, self.paperId_col, self.output_tf_col);

class TFVectorizorModel(Transformer):
    """
    Class that add a tf vector representation to each paper based on @paperId_col
    """

    def __init__(self, paper_profiles, paperId_col = "paper_id", output_tf_col="tf_vector"):
        """
        
        :param paper_profiles: a data frame that contains a tf profile of each paper
        :param paperId_col: name of the paper id column in paper_profiles as well as in the input data set of transform()
        :param output_tf_col: name of the result column that the model produces
        """
        # format (paper_id, tf_vector)
        self.paper_profiles = paper_profiles
        self.paperId_col = paperId_col
        self.output_tf_col = output_tf_col

    def setOutputTfCol(self, output_tf_col):
        self.paper_profiles = self.paper_profiles.withColumnRenamed(self.output_tf_col, output_tf_col)
        self.output_tf_col = output_tf_col

    def setPaperIdCol(self, paperId_col):
        self.paper_profiles = self.paper_profiles.withColumnRenamed(self.paperId_col, paperId_col)
        self.paperId_col = paperId_col

    def _transform(self, dataset):
        """
        Add for each paper, its corresponding tf vector.
        
        :param dataset: input data with a column "paper_id". Based on it, a tf vector for each paper is added.
        :return: data frame with additional column @output_tf_col
        """
        dataset = dataset.join(self.paper_profiles, self.paperId_col);
        return dataset

# TODO change comments and check it
class TFIDFVectorizer(Estimator):
    """
    Calculates a profile for each paper. Its representation is based on tf-idf scores.
    It is initialized with paper corpus - all papers taken into account when a representation
    of a paper is built.
    """

    def __init__(self, papers_corpus, paperId_col="paper_id", tf_map_col="term_occurrence", output_tf_idf_col="tf_idf_vector"):
        """
        Create an instance of the class.

        :param papers_corpus: data frame that contains all papers from the corpus. Format (paper_id, citeulike_paper_id).
        """
        self.papers_corpus = papers_corpus
        # data frame for a mapping between term_id and sequantial id based on all terms in the corpus
        self.term_mapping = None
        # data frame contains all the terms in the corpus and their corresponding ids
        self.term_corpus = None
        self.paperId_col = paperId_col
        self.tf_map_col = tf_map_col
        self.output_tf_idf_col = output_tf_idf_col

    def _fit(self, papers):
        """
        Build a tf-idf representation for each paper in the input data set. Based on papers in the papers corpus, a set of all
        terms is extracted. For each of them a unique id is generated. Term ids are sequential. Then depending on all terms
        and their frequence for a paper, a sparse vector is built. A model that can be used to map a tf-idf vector to each paper 
        based on its paper id is returned.

        :param dataset: input data set, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: a build model which can be used for transformation of a data set
        """

        # select only those papers that are part of the paper corpus
        papers = papers.join(self.papers_corpus, self.paperId_col)

        # explode map with key:value pairs
        exploded_papers = papers.select(self.paperId_col, F.explode(self.tf_map_col))
        # rename column:  key to term_id, value to tf (term frequence)
        exploded_papers = exploded_papers.withColumnRenamed("key", "term_id").withColumnRenamed("value", "tf")

        # collect all distinct term ids
        terms = exploded_papers.select("term_id").distinct()

        # generate sequential ids for terms
        term_corpus = terms.withColumn('indexed_term_id', F.row_number().over(Window.orderBy("term_id")))
        self.term_corpus = term_corpus

        # add document frequency for each term
        term_document_frequency = exploded_papers.groupBy("term_id").agg(F.count(exploded_papers.paper_id).alias("df"))
        exploded_papers = exploded_papers.join(term_document_frequency, "term_id")

        # join term corpus with exploded papers to add new id for each term
        # format (key, paper_id, value, id)
        indexed_exploded_papers = exploded_papers.join(term_corpus, "term_id")

        # collect (id, tf, df) pairs into one list
        # df document frequency
        indexed_exploded_papers = indexed_exploded_papers.groupby(self.paperId_col).agg(
            F.collect_list(F.struct("indexed_term_id", "tf", "df")).alias("term_occurrence"))

        voc_size = terms.count()
        papers_corpus_size = self.papers_corpus.count()
        dataset = indexed_exploded_papers.withColumn(self.output_tf_idf_col,
                                                     UDFContainer.getInstance().to_tf_idf_vector_udf("term_occurrence",
                                                                                                  F.lit(voc_size), F.lit(papers_corpus_size)))
        paper_profiles = dataset.select(self.paperId_col, self.output_tf_idf_col)
        return TFVectorizorModel(paper_profiles);


class TFIDFVectorizorModel(Transformer):
    """
    Class that add a tf-idf vector representation to each paper based on paper_id.
    """
    
    def __init__(self, paper_profiles, paperId_col="paper_id"):
        # format (paper_id, tf_idf_vector)
        self.paper_profiles = paper_profiles
        self.paperId_col = paperId_col

    def _transform(self, dataset):
        """
        Add for each paper, its corresponding tf-idf vector.

        :param dataset: input data with a column "paper_id". Based on it, a tf-idf vector for each paper is added.
        :return: data frame with additional column "tf_idf_vector"
        """
        dataset = dataset.join(self.paper_profiles, self.paperId_col);
        return dataset