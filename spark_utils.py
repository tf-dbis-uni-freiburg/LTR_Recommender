import datetime
import math
from collections import defaultdict
import pyspark.sql.functions as F
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
from pyspark.ml.linalg import VectorUDT
from random import randint, shuffle
import math
import scipy

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
        self.generate_peers = F.udf(UDFContainer.__generate_peers, ArrayType(IntegerType(), False))
        self.split_papers = F.udf(UDFContainer.__split_papers, ArrayType(ArrayType(StringType())))
        self.mrr_per_user = F.udf(UDFContainer.__mrr_per_user, DoubleType())
        self.ndcg_per_user = F.udf(UDFContainer.__ndcg_per_user, DoubleType())
        self.recall_per_user = F.udf(UDFContainer.__recall_per_user, DoubleType())
        self.get_candidate_set_per_user = F.udf(UDFContainer.__get_candidate_set_per_user, ArrayType(ArrayType(DoubleType())))
        self.calculate_prediction = F.udf(UDFContainer.__calculate_prediction, FloatType())

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

    def generate_peers_udf(self, positives, total_papers_count, k):
        """
        Generate peer papers for a paper. For example, if a total number of paper is 6, means that in the paper corpus these are the possible paper ids [1, 2, 3, 4, 5, 6].
        It randomly selects k of them. None of the selected id has to be in "positives" list.

        :param positives: list of paper ids. The intersection of the positives and the generated peers has to be empty.
        :param total_papers_count: total number of papers in the paper corpus
        :param k: how many peer papers have to be generated
        :return: a list of paper ids corresponding to peer papers for a paper
        """
        return self.generate_peers(positives, total_papers_count, k)

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

    def get_candidate_set_per_user_udf(self, total_predicted_papers, training_paper, k):
        """
        From a list of best predicted papers(k + max. number of papers for a user in the training set),
        remove those which a user already liked and they are included in the training set.

        :param total_predicted_papers list of tuples. Each contains (paper_id, prediction). Not sorted.
        Represents top predicted papers
        :param training_paper: all paper ids that are part of liked papers by a used in the training set
        :param k: how many top predictions have to be returned
        :return: top k predicted papers for a user
        """
        return self.get_candidate_set_per_user(total_predicted_papers, training_paper, k)


    def mrr_per_user_udf(self, predicted_papers, test_papers):
        """
        Calculate MRR for a specific user. Sort the predicted papers by prediction (DESC order).
        Find the first hit in the predicted papers and return 1/(index of the first hit). For
        example, if a test_paper is [7, 12, 19, 66, 10]. And sorted predicted_papers is 
        [(3, 5.5), (4, 4.5) , (5, 4.3), (7, 1.9), (12, 1.5), (10, 1.2), (66, 1.0)]. The first hit 
        is 7 which index is 3. Then the mrr is 1 / (3+1)
        
        :param predicted_papers: list of tuples. Each contains (paper_id, prediction). Not sorted.
        :param test_papers: list of paper ids. Each paper id is part of the test set for a user.
        :return: mrr 
        """
        return self.mrr_per_user(predicted_papers, test_papers)

    def recall_per_user_udf(self, predicted_papers, test_papers):
        """
        Calculate Recall for a specific user. Extract only paper ids from predicted_papers, discard prediction information.
        Then, find the number of hits (common paper ids) that both arrays have. Return (#hits)/ (size of test_papers). 
        For example, if a test_paper is [7, 12, 19, 66, 10]. And predicted_papers is  [(3, 5.5), (4, 4.5) , (5, 4.3), (7, 1.9), (12, 1.5), (10, 1.2), (66, 1.0)].
        Hits are [7, 12, 10, 66]. Then the result value is (4 / 5) = 0.8

        :param predicted_papers: list of tuples. Each contains (paper_id, prediction). Not sorted.
        :param test_papers: list of paper ids. Each paper id is part of the test set for a user.
        :return: recall
        """
        return self.recall_per_user(predicted_papers, test_papers)

    def ndcg_per_user_udf(self, predicted_papers, normalization_factor):
        """
        Calculate NDCG per user. Sort the predicted_papers by their predictions (DESC order). Then calculate
        DCG over the predicted papers. DCG is a sum over all predicted papers, for each predicted paper add
        ((prediction) / log(2, position of the paper in the list + 1)). And the end divide by normalization factor 
        - IDCG calculated over top k best predicted papers. For example, if sorted predicted_papers is 
        [(3, 5.5), (4, 4.5) , (5, 4.3)]. DCG will be ((5.5 / log2(2)) + (4.5 / log2(3)) + (4.3 / log2(4)))

        :param predicted_papers: list of tuples. Each contains (paper_id, prediction). Not sorted.
        :param normalization_factor: IDCG which is calculated as a sum over top k best predicted papers. Independent of a user.
        For each paper in the topk best papers, it is added ((2^prediction - 1)/(log(2, position of the paper in the list + 1))
        :return: NDCG for a user
        """
        return self.ndcg_per_user(predicted_papers, normalization_factor)

    def calculate_prediction_udf(self, features, coefficients):
        """
        Calculate a score prediction for a paper. Multiple its features vector
        with the coefficients received from the model.

        :param features: sparse vector, features vector of a paper
        :param coefficients: model coefficient, weights for each feature
        :return: prediction score 
        """
        return self.calculate_prediction(features, coefficients)

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

    def __generate_peers(positives, total_papers_count, k):
        """
        Generate peers papers for a paper. For example, if a total number of paper is 6, means that in the paper corpus these are the possible paper ids [1, 2, 3, 4, 5, 6].
        It randomly selects k of them. None of the selected id has to be in "positives" list.

        :param positives: list of paper ids. The intersection of the positives and the generated peers has to be empty.
        :param total_papers_count: total number of papers in the paper corpus
        :param k: how many peer papers have to be generated
        :return: a list of paper ids corresponding to peer papers for a paper
        """
        peers = set()
        while len(peers) < k:
            candidate = randint(1, total_papers_count + 1)
            # if a candidate paper is not in positives for an user and there exists paper with such id in the paper corpus
            if candidate not in positives and candidate not in peers:
                peers.add(candidate)
        return list(peers)

    def __to_tf_vector(terms_mapping, voc_size):
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

    def __to_tf_idf_vector(terms_mapping, terms_count, papers_count):
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
            tf_idf = tf * math.log(papers_count/df, 2)
            map[term_id] = tf_idf
        return Vectors.sparse(terms_count, map)

    def __split_papers(papers_id_list):
        """
        Shuffle the input list of paper ids and divide it into two lists. The ratio is 50/50.
        
        :param: papers_id_list initial list of paper ids that will be split
        :return: two arrays with paper ids. The first one contains the "positive paper ids" or those
        which difference will be added with label 1. The second - "the negative paper ids" -  added with
        label 0.
        """
        shuffle(papers_id_list)
        ratio = int(0.5 * len(papers_id_list))
        positive_class_set = papers_id_list[:ratio]
        negative_class_set = papers_id_list[ratio:]
        return [positive_class_set, negative_class_set]

    def __get_candidate_set_per_user(total_predicted_papers, training_paper, k):
        """
        From a list of best predicted papers(k + max. number of papers for a user in the training set),
        remove those which a user already liked and they are included in the training set.
        
        :param total_predicted_papers list of tuples. Each contains (paper_id, prediction). Not sorted.
        Represents top predicted papers
        :param training_paper: all paper ids that are part of liked papers by a used in the training set
        :param k: how many top predictions have to be returned
        :return: top k predicted papers for a user
        """
        # sort by prediction
        sorted_prediction_papers = sorted(total_predicted_papers, key=lambda tup: -tup[1])
        training_paper_set = set(training_paper)
        filtered_sorted_prediction_papers = [(float(x), float(y)) for x, y in sorted_prediction_papers if x not in training_paper_set]
        return filtered_sorted_prediction_papers[:k]

    def __mrr_per_user(predicted_papers, test_papers):
        """
        Calculate MRR for a specific user. Sort the predicted papers by prediction (DESC order).
        Find the first hit in the predicted papers and return 1/(index of the first hit). For
        example, if a test_paper is [7, 12, 19, 66, 10]. And sorted predicted_papers is 
        [(3, 5.5), (4, 4.5) , (5, 4.3), (7, 1.9), (12, 1.5), (10, 1.2), (66, 1.0)]. The first hit 
        is 7 which index is 3. Then the mrr is 1 / (3+1)

        :param predicted_papers: list of tuples. Each contains (paper_id, prediction). Not sorted.
        :param test_papers: list of paper ids. Each paper id is part of the test set for a user.
        :return: mrr 
        """
        # sort by prediction
        sorted_prediction_papers = sorted(predicted_papers, key=lambda tup: - tup[1])
        test_papers_set = set(test_papers)
        index = 1
        for i, prediction in sorted_prediction_papers:
            if (int(i) in test_papers_set):
                return 1 / index
            index += 1
        return 0.0

    def __recall_per_user(predicted_papers, test_papers):
        """
        Calculate Recall for a specific user. Extract only paper ids from predicted_papers, discard prediction information.
        Then, find the number of hits (common paper ids) that both arrays have. Return (#hits)/ (size of test_papers). 
        For example, if a test_paper is [7, 12, 19, 66, 10]. And predicted_papers is  [(3, 5.5), (4, 4.5) , (5, 4.3), (7, 1.9), (12, 1.5), (10, 1.2), (66, 1.0)].
        Hits are [7, 12, 10, 66]. Then the result value is (4 / 5) = 0.8
        
        :param predicted_papers: list of tuples. Each contains (paper_id, prediction). Not sorted.
        :param test_papers: list of paper ids. Each paper id is part of the test set for a user.
        :return: recall
        """
        predicted_papers = [int(x[0]) for x in predicted_papers]
        hits = set(predicted_papers).intersection(test_papers)
        return len(hits) / len(test_papers)

    def __ndcg_per_user(predicted_papers, normalization_factor):
        """
        Calculate NDCG per user. Sort the predicted_papers by their predictions (DESC order). Then calculate
        DCG over the predicted papers. DCG is a sum over all predicted papers, for each predicted paper add
        ((prediction) / log(2, position of the paper in the list + 1)). And the end divide by normalization factor 
        - IDCG calculated over top k best predicted papers. For example, if sorted predicted_papers is 
         [(3, 5.5), (4, 4.5) , (5, 4.3)]. DCG will be ((5.5 / log2(2)) + (4.5 / log2(3)) + (4.3 / log2(4)))
         
        :param predicted_papers: list of tuples. Each contains (paper_id, prediction). Not sorted.
        :param normalization_factor: IDCG which is calculated as a sum over top k best predicted papers. Independent of a user.
        For each paper in the topk best papers, it is added ((2^prediction - 1)/(log(2, position of the paper in the list + 1))
        :return: NDCG for a user
        """
        # sort by prediction
        sorted_prediction_papers = sorted(predicted_papers, key=lambda tup: -tup[1])
        i = 1
        sum = 0;
        for paper_id, prediction in sorted_prediction_papers:
            sum += prediction / (math.log2(i + 1))
            i += 1
        return  sum / normalization_factor

    def __calculate_prediction(features, coefficients):
        """
        Calculate a score prediction for a paper. Multiple its features vector
        with the coefficients received from the model.

        :param features: sparse vector, features vector of a paper
        :param coefficients: model coefficient, weights for each feature
        :return: prediction score 
        """
        cx = scipy.sparse.coo_matrix(features)
        prediction = 0.0
        for index, value in zip(cx.col, cx.data):
            prediction += value * coefficients[index]
        return float(prediction)


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