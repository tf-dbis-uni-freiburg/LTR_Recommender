from pyspark.ml.base import Estimator
from pyspark.ml.base import Transformer
from pyspark.ml.linalg import Vectors
from pyspark.ml.linalg import VectorUDT
import math
import pyspark.sql.functions as F
from pyspark.ml.clustering import LDA
from pyspark.sql.types import StructType, StructField, IntegerType

from logger import Logger


class TFVectorizer(Estimator):
    """
    Calculates a profile for each paper. Its representation is based on tf scores.
    It is initialized with paper corpus - all papers taken into account when a representation
    of a paper is built.
    """

    def __init__(self, papers_corpus, paperId_col="paper_id", tf_map_col="term_occurrence", output_col="tf_vector"):
        """
        Create an instance of the class.
        
        :param papers_corpus: PaperCorpus object that contains data frame. It represents all papers from the corpus. 
        Possible format (paper_id, citeulike_paper_id). See PaperCorpus documentation for more information.
        :param paperId_col name of the paper id column in the input data frame of fit()
        :param tf_map_col name of the tf representation column in the input data frame of fit(). The type of the 
        column is Map. It contains key:value pairs where key is the term id and value is #occurence of the term
        in a particular paper.
        :param output_col the name of the column in which the produced result is stored - tf vector of a paper
        """
        self.papers_corpus = papers_corpus
        # data frame contains all the terms in the corpus and their corresponding ids
        self.term_corpus = None
        self.paperId_col= paperId_col
        self.tf_map_col = tf_map_col
        self.output_col = output_col

    def setPaperIdCol(self, paperId_col):
        self.paperId_col = paperId_col

    def setTFMapCol(self, tf_map_col):
        self.tf_map_col = tf_map_col

    def setOutputTFCol(self, output_col):
        self.output_col

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
        # term_occurrence
        exploded_papers = papers.select(self.paperId_col, F.explode(self.tf_map_col)).withColumnRenamed("key", "term_id")
        # collect all distinct term ids
        terms = exploded_papers.select("term_id").distinct()

        # generate sequential ids for terms, use zipWithIndex to generate ids starting from 0
        # (name, dataType, nullable)
        term_corpus_schema = StructType([StructField("term_id", IntegerType(), False),
                                         StructField("id", IntegerType(), False)])
        term_corpus = terms.rdd.zipWithIndex().map(lambda x: (int(x[0][0]), x[1])).toDF(term_corpus_schema)
        self.term_corpus = term_corpus

        # join term corpus with exploded papers to add new id for each term
        # format (key, paper_id, value, id)
        indexed_exploded_papers = exploded_papers.join(term_corpus, "term_id")

        # collect (id, value) pairs into one list
        indexed_exploded_papers = indexed_exploded_papers.groupby(self.paperId_col).agg(F.collect_list(F.struct("id", "value")).alias("term_occurrence"))
        voc_size = terms.count()

        def to_tf_vector(terms_mapping, voc_size):
            """
            From a list of lists ([[], [], [], ...]) create a sparse vector. Each sublist contains 2 values.
            The first is the term id. The second is the number of occurences, the term appears in a paper.

            :param terms_mapping: a list of lists. Each sublist contains two elements.
            :param voc_size: the size of returned Sparse vector, total number of terms in the paper corpus
            :return: sparse vector based on the input mapping. It is a tf representation of a paper
            """
            map = {}
            for term_id, term_occurrence in terms_mapping:
                map[term_id] = term_occurrence
            return Vectors.sparse(voc_size, map)


        to_tf_vector_udf = F.udf(to_tf_vector, VectorUDT())
        dataset = indexed_exploded_papers.withColumn(self.output_col, to_tf_vector_udf("term_occurrence", F.lit(voc_size)))
        paper_profiles = dataset.select(self.paperId_col, self.output_col)
        return TFVectorizorModel(paper_profiles, self.paperId_col, self.output_col);

class TFVectorizorModel(Transformer):
    """
    Class that add a tf vector representation to each paper based on @paperId_col
    """

    def __init__(self, paper_profiles, paperId_col = "paper_id", output_col="tf_vector"):
        """
        Build a tf-vectorizer model. It adds a tf-vector representation to each paper based on its paper id.
        It is stored in paperId_col. The result the model produces is stored in output_col.
        
        :param paper_profiles: a data frame that contains a tf profile of each paper
        :param paperId_col: name of the paper id column in the input data set of transform()
        :param output_col: name of the result column that the model produces
        """
        # format (paper_id, tf_vector)
        self.paper_profiles = paper_profiles
        self.paperId_col = paperId_col
        self.output_col = output_col

    def setOutputTfCol(self, output_col):
        """
        Change the name of the column in which the result of the transform operation is stored. 
        
        :param output_col: new name of the result column that the model produces
        """
        self.paper_profiles = self.paper_profiles.withColumnRenamed(self.output_col, output_col)
        self.output_col = output_col

    def setPaperIdCol(self, paperId_col):
        """
        Change the name of the column in which a paper id is stored. Based on it a tf representation
        to each paper id added.

        :param paperId_col: new name of the paper id column in the input data set of transform()
        """
        self.paper_profiles = self.paper_profiles.withColumnRenamed(self.paperId_col, paperId_col)
        self.paperId_col = paperId_col

    def _transform(self, dataset):
        """
        Add for each paper, its corresponding tf vector.
        
        :param dataset: input data with a column paperId_col. Based on it, a tf vector for each paper is added.
        :return: data frame with additional column output_col
        """
        dataset = dataset.join(self.paper_profiles, self.paperId_col);
        return dataset

class TFIDFVectorizer(Estimator):
    """
    Calculates a profile for each paper. Its representation is based on tf-idf scores.
    It is initialized with paper corpus - all papers taken into account when a representation
    of a paper is built.
    """

    def __init__(self, papers_corpus, paperId_col="paper_id", tf_map_col="term_occurrence", output_col="tf_idf_vector"):
        """
        Create an instance of the class.

        :param papers_corpus: data frame that contains all papers from the corpus. Format (paper_id, citeulike_paper_id).
        :param paperId_col name of the paper id column in the input data frame of fit()
        :param tf_map_col name of the tf representation column in the input data frame of fit(). The type of the 
        column is Map. It contains key:value pairs where key is the term id and value is #occurence of the term
        in a particular paper.
        :param output_col the name of the column in which the produced result is stored - tf-idf vector of a paper
        """
        self.papers_corpus = papers_corpus
        # data frame contains all the terms in the corpus and their corresponding ids
        self.term_corpus = None
        self.paperId_col = paperId_col
        self.tf_map_col = tf_map_col
        self.output_col = output_col

    def _fit(self, bag_of_words):
        """
        Build a tf-idf representation for each paper in the input data set. Based on papers in the papers corpus, a set of all
        terms is extracted. For each of them a unique id is generated. Term ids are sequential. Then depending on all terms
        and their frequence for a paper, a sparse vector is built. A model that can be used to map a tf-idf vector to each paper 
        based on its paper id is returned.

        :param bag_of_words: input data set, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: a build model which can be used for transformation of a data set
        """

        # select only those papers that are part of the paper corpus
        papers = bag_of_words.join(self.papers_corpus.papers, bag_of_words[self.paperId_col] == self.papers_corpus.papers[self.papers_corpus.paperId_col]).drop(bag_of_words[self.paperId_col])

        # explode map with key:value pairs
        exploded_papers = papers.select(self.paperId_col, F.explode(self.tf_map_col))
        # rename column:  key to term_id, value to tf (term frequence)
        exploded_papers = exploded_papers.withColumnRenamed("key", "term_id").withColumnRenamed("value", "tf")

        # collect all distinct term ids
        terms = exploded_papers.select("term_id").distinct()

        # generate sequential ids for terms, use zipWithIndex to generate ids starting from 0
        # (name, dataType, nullable)
        term_corpus_schema = StructType([StructField("term_id", IntegerType(), False),
                                     StructField("indexed_term_id", IntegerType(), False)])
        term_corpus = terms.rdd.zipWithIndex().map(lambda x: (int(x[0][0]), x[1])).toDF(term_corpus_schema)
        self.term_corpus = term_corpus

        # add document frequency for each term
        term_document_frequency = exploded_papers.groupBy("term_id").agg(F.count(exploded_papers.paper_id).alias("df"))
        exploded_papers = exploded_papers.join(term_document_frequency, "term_id")

        # join term corpus with exploded papers to add new id for each term
        # format (term_id, paper_id, tf, df, indexed_term_id)
        indexed_exploded_papers = exploded_papers.join(term_corpus, "term_id")

        # collect (id, tf, df) pairs into one list
        # df document frequency
        indexed_exploded_papers = indexed_exploded_papers.groupby(self.paperId_col).agg(
            F.collect_list(F.struct("indexed_term_id", "tf", "df")).alias("term_occurrence"))

        voc_size = terms.count()
        papers_corpus_size = self.papers_corpus.papers.count()

        def to_tf_idf_vector(terms_mapping, terms_count, papers_count):
            """
            From a list of lists ([[], [], [], ...]) create a sparse vector. Each sublist contains 3 values.
            The first is the term id. The second is the number of occurrences, the term appears in a paper - its
            term frequence. The third is the number of papers the term appears - its document frequency.

            :param terms_mapping: a list of lists. Each sublist contains three elements.
            :param terms_count: the size of returned Sparse vector, total number of terms in the paper corpus
            :param papers_count: total number of papers in the corpus
            :return: sparse vector based on the input mapping. It is a tf-idf representation of a paper
            """
            map = {}
            for term_id, tf, df in terms_mapping:
                tf_idf = tf * math.log(papers_count / df, 2)
                map[term_id] = tf_idf
            return Vectors.sparse(terms_count, map)


        to_tf_idf_vector_udf = F.udf(to_tf_idf_vector, VectorUDT())

        dataset = indexed_exploded_papers.withColumn(self.output_col, to_tf_idf_vector_udf("term_occurrence", F.lit(voc_size), F.lit(papers_corpus_size)))
        paper_profiles = dataset.select(self.paperId_col, self.output_col)
        return TFIDFVectorizorModel(paper_profiles, self.paperId_col, self.output_col);


class TFIDFVectorizorModel(Transformer):
    """
    Class that add a tf-idf vector representation to each paper based on paperId_col.
    """
    
    def __init__(self, paper_profiles, paperId_col="paper_id", output_col="tf_idf_vector"):
        """
        Build a tfidf-vectorizer model. It adds a tfidf-vector representation to each paper based on its paper id.
        It is stored in paperId_col. The result the model produces is stored in output_col.

        :param paper_profiles: a data frame that contains a tf profile of each paper
        :param paperId_col: name of the paper id column in the input data set of transform()
        :param output_col: name of the result column that the model produces
        """
        # format (paper_id, tf_idf_vector)
        self.paper_profiles = paper_profiles
        self.paperId_col = paperId_col
        self.output_col = output_col

    def setOutputCol(self, output_col):
        """
        Change the name of the column in which the result of the transform operation is stored. 

        :param output_col: new name of the result column that the model produces
        """
        self.paper_profiles = self.paper_profiles.withColumnRenamed(self.output_col, output_col)
        self.output_col = output_col

    def setPaperIdCol(self, paperId_col):
        """
        Change the name of the column in which a paper id is stored. Based on it a tf representation
        to each paper id added.

        :param paperId_col: new name of the paper id column in the input data set of transform()
        """
        self.paper_profiles = self.paper_profiles.withColumnRenamed(self.paperId_col, paperId_col)
        self.paperId_col = paperId_col

    def _transform(self, dataset):
        """
        Add for each paper, its corresponding tf-idf vector.

        :param dataset: input data with a column paperId_col. Based on it, a tf-idf vector for each paper is added.
        :return: data frame with additional column output_col
        """
        dataset = dataset.join(self.paper_profiles, self.paperId_col);
        return dataset

class LDAVectorizer(Estimator):


    def __init__(self, papers_corpus, k_topics = 5, maxIter = 10, paperId_col = "paper_id", tf_map_col = "term_occurrence", output_col = "lda_vector"):
        """
        Create an instance of the class.
    
        :param papers_corpus: PaperCorpus object that contains data frame. It represents all papers from the corpus. 
        Possible format (paper_id, citeulike_paper_id). See PaperCorpus documentation for more information.
        :param k_topics the number of topics (clusters) to infer. Must be > 1
        :param maxIter max number of iterations (>= 0)
        :param paperId_col name of the paper id column in the input data frame of fit()
        :param tf_map_col name of the tf representation column in the input data frame of fit(). The type of the 
        column is Map. It contains key:value pairs where key is the term id and value is #occurence of the term
        in a particular paper.
        :param output_col the name of the column in which the produced result is stored - lda vector of a paper
        """
        self.papers_corpus = papers_corpus
        # data frame contains all the terms in the corpus and their corresponding ids
        self.term_corpus = None
        self.k_topics = k_topics
        self.maxIter = maxIter
        self.paperId_col = paperId_col
        self.tf_map_col = tf_map_col
        self.output_col = output_col


    def setPaperIdCol(self, paperId_col):
        self.paperId_col = paperId_col


    def setTFMapCol(self, tf_map_col):
        self.tf_map_col = tf_map_col


    def setOutputTFCol(self, output_col):
        self.output_col


    def _fit(self, papers):
        """
        Build a LDA representation for each paper in the input data set. Based on papers in the papers corpus, a set of all
        terms is extracted. For each of them a unique id is generated. Term ids are sequential. Then depending on all terms
        and their frequence for a paper, a sparse vector is built. A model that can be used to map a tf vector to each paper 
        based on its paper id is used. Based on the tf representation of all papers - LDA is trained and used for prodicing
        LDA representation.
    
        :param data set: input data set, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: a build model which can be used for transformation of a data set
        """
        Logger.log("Train/Transform TF vectorizer.")
        tfVectorizer = TFVectorizer(self.papers_corpus, paperId_col = self.paperId_col, tf_map_col = self.tf_map_col, output_col = "tf_vector")
        tfVectorizerModel = tfVectorizer.fit(papers)
        # paper_id | tf_vector
        papers_tf_vectors = tfVectorizerModel.transform(papers).select(self.paperId_col, "tf_vector")
        papers_tf_vectors.cache()
        Logger.log("Train LDA. Topics:" + str(self.k_topics))
        # Trains a LDA model.
        # The number of topics to infer. Must be > 1.
        lda = LDA(featuresCol = "tf_vector", k = self.k_topics)
        model = lda.fit(papers_tf_vectors)

        Logger.log("Transform LDA over paper corpus.")
        # paper_id | lda_vector
        papers_lda_vectors = model.transform(papers_tf_vectors).withColumnRenamed("topicDistribution", self.output_col).drop("tf_vector")

        Logger.log("Return LDA model.")
        papers_tf_vectors.unpersist()
        return LDAModel(papers_lda_vectors, self.paperId_col, self.output_col);

class LDAModel(Transformer):
    """
    Class that add a LDA vector representation to each paper based on @paperId_col
    """

    def __init__(self, paper_profiles, paperId_col = "paper_id", output_col = "lda_vector"):
        """
        Build a LDA-vectorizer model. It adds a lda-vector representation to each paper based on its paper id.
        It is stored in paperId_col. The result the model produces is stored in output_col.

        :param paper_profiles: a data frame that contains a tf profile of each paper
        :param paperId_col: name of the paper id column in the input data set of transform()
        :param output_col: name of the result column that the model produces
        """
        # format (paper_id, lda_vector)
        self.paper_profiles = paper_profiles
        self.paperId_col = paperId_col
        self.output_col = output_col

    def setOutputCol(self, output_col):
        """
        Change the name of the column in which the result of the transform operation is stored. 

        :param output_col: new name of the result column that the model produces
        """
        self.paper_profiles = self.paper_profiles.withColumnRenamed(self.output_col, output_col)
        self.output_col = output_col

    def setPaperIdCol(self, paperId_col):
        """
        Change the name of the column in which a paper id is stored. Based on it a lda representation
        to each paper id added.

        :param paperId_col: new name of the paper id column in the input data set of transform()
        """
        self.paper_profiles = self.paper_profiles.withColumnRenamed(self.paperId_col, paperId_col)
        self.paperId_col = paperId_col

    def _transform(self, dataset):
        """
        Add for each paper, its corresponding LDA vector.

        :param dataset: input data with a column paperId_col. Based on it, a lda vector for each paper is added.
        :return: data frame with additional column output_col
        """
        Logger.log("LTR LDA model transform method called.")
        dataset = dataset.join(self.paper_profiles, self.paperId_col);
        return dataset

