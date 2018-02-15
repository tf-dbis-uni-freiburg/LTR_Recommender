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

    def __init__(self, spark, papers_corpus):
        """
        Create an instance of the class.
        
        :param papers_corpus: data frame that contains all papers from the corpus. Format (paper_id, citeulike_paper_id).
        """
        self.papers_corpus = papers_corpus
        # data frame for a mapping between term_id and sequantial id based on all terms in the corpus
        self.term_mapping = None
        # data frame contains all the terms in the corpus and their corresponding ids
        self.term_corpus = None

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
        papers = papers.join(self.papers_corpus, "paper_id")

        # explode map with key:value pairs
        exploded_papers = papers.select("paper_id", F.explode("term_occurrence"))

        # collect all distinct term ids
        terms = exploded_papers.select("key").distinct()

        # generate sequential ids for terms
        term_corpus = terms.withColumn('id', F.row_number().over(Window.orderBy("key")))
        self.term_corpus = term_corpus

        # join term corpus with exploded papers to add new id for each term
        # format (key, paper_id, value, id)
        indexed_exploded_papers = exploded_papers.join(term_corpus, "key")
        # collect (id, value) pairs into one list
        indexed_exploded_papers = indexed_exploded_papers.groupby("paper_id").agg(F.collect_list(F.struct("id", "value")).alias("term_occurrence"))
        voc_size = terms.count()

        dataset = indexed_exploded_papers.withColumn("tf_vector", UDFContainer.getInstance().to_tf_vector_udf("term_occurrence", F.lit(voc_size)))

        paper_profiles = dataset.select("paper_id", "tf_vector")
        return TFVectorizorModel(paper_profiles);

class TFVectorizorModel(Transformer):
    """
    Class that add a tf vector representation to each paper based on paper_id.
    """

    def __init__(self, paper_profiles,):
        # format (paper_id, tf_vector)
        self.paper_profiles = paper_profiles

    def _transform(self, dataset):
        """
        Add for each paper, its corresponding tf vector.
        
        :param dataset: input data with a column "paper_id". Based on it, a tf vector for each paper is added.
        :return: data frame with additional column "tf_vector"
        """
        dataset = dataset.join(self.paper_profiles, "paper_id");
        return dataset


class TFIDFVectorizer(Estimator):
    """
    Calculates a profile for each paper. Its representation is based on tf-idf scores.
    It is initialized with paper corpus - all papers taken into account when a representation
    of a paper is built.
    """

    def __init__(self, spark, papers_corpus):
        """
        Create an instance of the class.

        :param papers_corpus: data frame that contains all papers from the corpus. Format (paper_id, citeulike_paper_id).
        """
        self.papers_corpus = papers_corpus
        # data frame for a mapping between term_id and sequantial id based on all terms in the corpus
        self.term_mapping = None
        # data frame contains all the terms in the corpus and their corresponding ids
        self.term_corpus = None

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
        papers = papers.join(self.papers_corpus, "paper_id")

        # explode map with key:value pairs
        exploded_papers = papers.select("paper_id", F.explode("term_occurrence"))
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
        indexed_exploded_papers = indexed_exploded_papers.groupby("paper_id").agg(
            F.collect_list(F.struct("indexed_term_id", "tf", "df")).alias("term_occurrence"))

        voc_size = terms.count()

        papers_corpus_size = self.papers_corpus.count()
        dataset = indexed_exploded_papers.withColumn("tf_idf_vector",
                                                     UDFContainer.getInstance().map_to_vector_udf("term_occurrence",
                                                                                                  F.lit(voc_size), F.lit(papers_corpus_size)))

        paper_profiles = dataset.select("paper_id", "tf_idf_vector")
        return TFVectorizorModel(paper_profiles);


class TFIDFVectorizorModel(Transformer):
    """
    Class that add a tf-idf vector representation to each paper based on paper_id.
    """

    def __init__(self, paper_profiles, ):
        # format (paper_id, tf_idf_vector)
        self.paper_profiles = paper_profiles

    def _transform(self, dataset):
        """
        Add for each paper, its corresponding tf-idf vector.

        :param dataset: input data with a column "paper_id". Based on it, a tf-idf vector for each paper is added.
        :return: data frame with additional column "tf_idf_vector"
        """
        dataset = dataset.join(self.paper_profiles, "paper_id");
        return dataset