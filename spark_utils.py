import datetime
import math
from collections import defaultdict
import pyspark.sql.functions as F
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
from pyspark.ml.linalg import VectorUDT
from random import randint, shuffle

class UDFContainer():
    """
    Container of custom UDF functions. When the class is initialized, all custom UDF functions are registered.
    """
    _instance = None

    def __init__(self):
        # register all udfs when container is initialized
        self.vector_diff = F.udf(UDFContainer.__diff, VectorUDT())
        self.to_tf_vector = F.udf(UDFContainer.__to_tf_vector, VectorUDT())
        self.to_tf_idf_vector = F.udf(UDFContainer.__to_tf_idf_vector, VectorUDT())
        self.generate_negatives = F.udf(UDFContainer.__generate_negatives, ArrayType(IntegerType(), False))
        self.split_papers = F.udf(UDFContainer.__split_papers, ArrayType(ArrayType(StringType())))

    @staticmethod
    def getInstance():
        """
        Return an instance of the class. Similar to Singleton pattern. The first time, an instance is requested, it will be initialized.
        The other times after that the same initial instance will be returned.

        :return: instance of the class
        """
        if (UDFContainer._instance == None):
            UDFContainer._instance = UDFContainer()
        return UDFContainer._instance

    ### Methods for accessing the registered UDF functions ###

    def vector_diff_udf(self, v1, v2):
        """
        Calculate the difference between two sparse vectors.

        :return: sparse vector of their difference
        """
        return self.vector_diff(v1, v2)

    def generate_negatives_udf(self, positives, total_papers_count, k):
        """
        Generate negative papers for a paper. For example, if a total number of paper is 6, means that in the paper corpus these are the possible paper ids [1, 2, 3, 4, 5, 6].
        It randomly selects k of them. None of the selected id has to be in "positives" list.

        :param positives: list of paper ids. The intersection of the positives and the generated negatives has to be empty.
        :param total_papers_count: total number of papers in the paper corpus
        :param k: how many negative papers have to be generated
        :return: a list of paper ids corresponding to negative papers for a paper
        """
        return self.generate_negatives(positives, total_papers_count, k)

    def to_tf_vector_udf(self, terms_mapping, voc_size):
        """
        From a list of lists ([[], [], [], ...]) create a sparse vector. Each sublist contains 2 values.
        The first is the term id. The second is the number of occurences, the term appears in a paper.
        
        :param terms_mapping: a list of lists. Each sublist contains 2 values
        :param voc_size: the size of returned Sparse vector, total number of terms in the paper corpus
        :return: sparse vector based on the input mapping. It is a tf representation of a paper
        """
        return self.to_tf_vector(terms_mapping, voc_size)


    def to_tf_idf_vector_udf(self, terms_mapping, voc_size, papers_count):
        """
        From a list of lists ([[], [], [], ...]) create a sparse vector. Each sublist contains 3 values.
        The first is the term id. The second is the number of occurences, the term appears in a paper - its
        term frequence. The third is the number of papers the term appears - its document frequency.

        :param terms_mapping: a list of lists. Each sublist contains 3 values
        :param terms_count: the size of returned Sparse vector, total number of terms in the paper corpus
        :param papers_count: total number of papers in the corpus
        :return: sparse vector based on the input mapping. It is a tf-idf representation of a paper
        """
        return self.to_tf_idf_vector(terms_mapping, voc_size, papers_count)

    def split_papers_udf(self, papers_id_list):
        """
        Shuffle the input list of paper ids and divide it into two lists. The ratio is 50/50.
        
        :param: papers_id_list initial list of paper ids that will be split
        :return: two arrays with paper ids. The first one contains the "positive paper ids" or those
        which difference will be added with label 1. The second - "the negative paper ids" -  added with
        label 0.
        """
        return self.split_papers(papers_id_list)

    ### Private Functions ###

    def __build_publication_date(year, month):
        """
        Build a date based on input year and month. The month value can be null.
        """
        if (month == None):
            month = "jan"
        # convert month name to month number
        month_number = datetime.datetime.strptime(month, '%b').month
        # always from the first day of the month
        row_date = datetime.datetime(int(year), month_number, 1)
        return row_date

    def __diff(v1, v2):
        """
        Calculate the difference between two sparse vectors.

        :return: sparse vector of their difference
        """
        values = defaultdict(float)  # Dictionary with default value 0.0
        # Add values from v1
        for i in range(v1.indices.size):
            values[v1.indices[i]] += v1.values[i]
        # subtract values from v2
        for i in range(v2.indices.size):
            values[v2.indices[i]] -= v2.values[i]
        return Vectors.sparse(v1.size, dict(values))

    def __generate_negatives(positives, total_papers_count, k):
        """
        Generate negative papers for a paper. For example, if a total number of paper is 6, means that in the paper corpus these are the possible paper ids [1, 2, 3, 4, 5, 6].
        It randomly selects k of them. None of the selected id has to be in "positives" list.

        :param positives: list of paper ids. The intersection of the positives and the generated negatives has to be empty.
        :param total_papers_count: total number of papers in the paper corpus
        :param k: how many negative papers have to be generated
        :return: a list of paper ids corresponding to negative papers for a paper
        """
        negatives = set()
        while len(negatives) < k:
            candidate = randint(1, total_papers_count + 1)
            # if a candidate paper is not in positives for an user and there exists paper with such id in the paper corpus
            if candidate not in positives and candidate not in negatives:
                negatives.add(candidate)
        return list(negatives)

    def __to_tf_vector(terms_mapping, voc_size):
        """
        From a list of lists ([[], [], [], ...]) create a sparse vector. Each sublist contains 2 values.
        The first is the term id. The second is the number of occurences, the term appears in a paper.
        
        :param terms_mapping:
        :param voc_size: the size of returned Sparse vector, total number of terms in the paper corpus
        :return: sparse vector based on the input mapping. It is a tf representation of a paper
        """
        map = {}
        for term_id, term_occurrence in terms_mapping:
            map[term_id] = term_occurrence
        # mapping of terms starts from 1, compensate with the length of the vector
        return Vectors.sparse(voc_size + 1, map)

    def __to_tf_idf_vector(terms_mapping, terms_count, papers_count):
        """
        From a list of lists ([[], [], [], ...]) create a sparse vector. Each sublist contains 3 values.
        The first is the term id. The second is the number of occurences, the term appears in a paper - its
        term frequence. The third is the number of papers the term appears - its document frequency.

        :param terms_mapping: a list of lists. Each sublist contains three elements.
        :param terms_count: the size of returned Sparse vector, total number of terms in the paper corpus
        :param papers_count: total number of papers in the corpus
        :return: sparse vector based on the input mapping. It is a tf-idf representation of a paper
        """
        map = {}
        for term_id, tf, df in terms_mapping:
            tf_idf = tf * math.log(papers_count/df, 2)
            map[term_id] = tf_idf
        # mapping of terms starts from 1, compensate with the length of the vector
        return Vectors.sparse(terms_count + 1, map)

    def __split_papers(papers_id_list):
        """
        Shuffle the input list of paper ids and divide it into two lists. The ratio is 50/50.
        
        :param: papers_id_list initial list of paper ids that will be split
        :return: two arrays with paper ids. The first one contains the "positive paper ids" or those
        which difference will be added with label 1. The second - "the negative paper ids" -  added with
        label 0.
        """
        print("callinh")
        shuffle(papers_id_list)
        ratio = int(0.5 * len(papers_id_list))
        positive_class_set = papers_id_list[:ratio]
        print(positive_class_set)
        negative_class_set = papers_id_list[ratio:]
        print(negative_class_set)
        return [positive_class_set, negative_class_set]

class SparkBroadcaster():
    """
    Class for broadcasting variables.
    """
    _instance = None

    def __init__(self, spark):
        self.spark = spark

    @staticmethod
    def initialize(spark):
        """
        Create a single instance of the class
        
        :param spark: spark instance used for broadcasting variables
        :return: the instance of the class
        """
        SparkBroadcaster._instance = SparkBroadcaster(spark)

    @staticmethod
    def getInstance():
        """
        Return an instance of the class. Similar to Singleton pattern. 

        :return: instance of the class
        """
        return SparkBroadcaster._instance

    # Example of possible broadcasting
    # def broadcastTermsMapping(self, terms_mapping):
    #     global broadcasted_terms_mapping
    #     broadcasted_terms_mapping = self.spark.sparkContext.broadcast(terms_mapping)
    #
    # def getBroadcastedTermsMapping(self):
    #     return broadcasted_terms_mapping