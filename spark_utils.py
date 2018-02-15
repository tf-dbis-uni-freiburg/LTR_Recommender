import datetime
from collections import defaultdict
import pyspark.sql.functions as F
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
from pyspark.ml.linalg import VectorUDT
from random import randint

class UDFContainer():
    """
    Container of custom UDF functions. When the class is initialized, all custom UDF functions are registered.
    """
    _instance = None

    def __init__(self):
        # register all udfs when container is initialized
        self.vector_diff = F.udf(UDFContainer.__diff, VectorUDT())
        self.map_to_vector = F.udf(UDFContainer.__map_to_vector, VectorUDT())
        self.generate_negatives = F.udf(UDFContainer.__generate_negatives, ArrayType(IntegerType(), False))

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

    def map_to_vector_udf(self, terms_mapping, voc_size):
        """
        From a list of lists ([[], [], [], ...]) create a sparse vector. Each sublist contains 2 values.
        The first is the term id. The second is the number of occurences, the term appears in a paper.

        :param voc_size: the size of returned Sparse vector, total number of terms in the paper corpus
        :return: sparse vector based on the input mapping. It is a tf representation of a paper
        """
        return self.map_to_vector(terms_mapping, voc_size)

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

    def __map_to_vector(terms_mapping, size):
        """
        From a list of lists ([[], [], [], ...]) create a sparse vector. Each sublist contains 2 values.
        The first is the term id. The second is the number of occurences, the term appears in a paper.
        
        :param voc_size: the size of returned Sparse vector, total number of terms in the paper corpus
        :return: sparse vector based on the input mapping. It is a tf representation of a paper
        """
        map = {}
        for term_id, term_occurrence in terms_mapping:
            map[term_id] = term_occurrence
        # mapping of terms starts from 1, conpensate with the length of the vector
        return Vectors.sparse(size + 1, map)

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