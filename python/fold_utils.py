from pyspark.sql import functions as F
import csv
from pyspark.sql import Row
from spark_utils import UDFContainer
from logger import Logger
from paper_corpus_builder import PaperCorpusBuilder, PapersCorpus
from vectorizers import *
from random import shuffle
from learning_to_rank_spark2 import *
from pyspark.sql.types import *
from dateutil.relativedelta import relativedelta
from pyspark.sql.window import Window
import datetime
import numpy as np
from pyspark.ml.clustering import KMeans
import os
class Fold:
    """
    Encapsulates the notion of a fold. Each fold consists of training, test data frame and papers corpus. Each fold has an index which indicated 
    its position in the sequence of all extracted folds. The index starts from 1. Each fold is extracted based on the timestamp of 
    the samples. The samples in the test set are from a particular period in time. For example, the test_set can contains samples from
    [2008-09-29, 2009-03-29] and the training_set from [2004-11-04, 2008-09-28]. Important note is that the test set starts when the training 
    set ends. The samples are sorted by their timestamp column. Therefore, additional information for each fold is when the test set starts and ends. 
    Respectively the same for the training set. Also, the duration of the test set - period_in_months.  Papers corpus for each fold contains all the papers 
    published before the end date of a test set in the fold.
    """

    """ Name of the file in which test data frame of a fold is stored. """
    TEST_DF_CSV_FILENAME = "test.csv"
    """ Name of the file in which training data frame of a fold is stored. """
    TRAINING_DF_CSV_FILENAME = "training.csv"
    """ Name of the file in which papers corpus of a fold is stored. """
    PAPER_CORPUS_DF_CSV_FILENAME = "papers-corpus.csv"
    """ Name of the file in which LDA data frame of a fold is stored. """
    LDA_DF_FILENAME = "lda-papers.parquet"
    """ Name of the file in which candidate set data frame of a fold is stored. """
    CANDIDATE_DF_FILENAME = "candidate-set.parquet"
    """ Name of the file in which information for user clusters for a fold is stored. """
    USER_CLUSTERS_DF_FILENAME = "user-cluster.parquet"
    """ Name of the file in which overall results are written. """
    RESULTS_CSV_FILENAME = "results.csv"
    """ Prefix of the name of the folder in which the fold is stored in distributed manner. """
    DISTRIBUTED_PREFIX_FOLD_FOLDER_NAME = "distributed-fold-"
    """ Prefix of the name of the folder in which the fold is stored. """
    PREFIX_FOLD_FOLDER_NAME = "fold-"

    def __init__(self, training_data_frame, test_data_frame, output_path, split_method='time-aware'):
        self.index = None
        self.training_data_frame = training_data_frame
        self.test_data_frame = test_data_frame
        self.period_in_months = None
        self.tr_start_date = None
        self.ts_end_date = None
        self.ts_start_date = None
        self.papers_corpus = None
        self.folder_path = None
        self.ldaModel = None
        # user_id, list of candidate papers
        self.candidate_set = None
        # cluster_id, centroid, [user_ids] a list of users that belong to this cluster
        self.user_clusters = None
        self.output_path = output_path
        self.split_method = split_method

    def set_index(self, index):
        self.index = index

    def set_papers_corpus(self, papers_corpus):
        self.papers_corpus = papers_corpus

    def set_period_in_months(self, period_in_months):
        self.period_in_months = period_in_months

    def set_test_set_start_date(self, ts_start_date):
        self.ts_start_date = ts_start_date

    def set_test_set_end_date(self, ts_end_date):
        self.ts_end_date = ts_end_date

    def set_training_set_start_date(self, tr_start_date):
        self.tr_start_date = tr_start_date

    def store_distributed(self):
        """
        For a fold, store its test data frame, training data frame, its papers corpus, lda profiles for all
        paper in its paper corpus and candidate set of papers for each user in the test set.
        All of them are stored in a folder which name is based on PREFIX_FOLD_FOLDER_NAME and the index
        of a fold. For example, for a fold with index 2, the stored information for it will be in
        "distributed-fold-2" folder.
        """
        # save test data frame
        self.test_data_frame.write.csv(os.path.join(self.output_path,Fold.get_test_data_frame_path(self.index)))
        # save training data frame
        self.training_data_frame.write.csv(os.path.join(self.output_path,Fold.get_training_data_frame_path(self.index)))
        # save paper corpus
        self.papers_corpus.papers.write.csv(os.path.join(self.output_path,Fold.get_papers_corpus_frame_path(self.index)))
        # save lda paper profiles
        # parquet used because we cannot store vectors (lda vectors) in csv format
        self.ldaModel.paper_profiles.write.parquet(os.path.join(self.output_path,Fold.get_lda_papers_frame_path(self.index)))
        # save candidate set for each paper in the test set
        # format - user_id, [candidate_set]
        # parquet used because we cannot store vectors (lda vectors) in csv format
        self.candidate_set.write.parquet(os.path.join(self.output_path,Fold.get_candidate_set_data_frame_path(self.index)))
        # save user clusters
        # format - cluster_id, centroid, [user_ids]
        # parquet used because we cannot store vectors (lda vectors) in csv format
        # self.user_clusters.write.parquet(Fold.get_user_clusters_data_frame_path(self.index))

    def store(self):
        """
        For a fold, store its test data frame, training data frame, its papers corpus, lda profiles for all
        paper in its paper corpus and candidate set of papers for each user in the test set.
        Each data frame will be stored in a single csv file.
        All of them are stored in a folder which name is based on PREFIX_FOLD_FOLDER_NAME and the index
        of a fold. For example, for a fold with index 2, the stored information for it will be in
        "fold-2" folder.
        """
        # save test data frame
        self.test_data_frame.coalesce(1).write.csv(os.path.join(self.output_path,Fold.get_test_data_frame_path(self.index, distributed=False)))
        # save training data frame
        self.training_data_frame.coalesce(1).write.csv(os.path.join(self.output_path,Fold.get_training_data_frame_path(self.index, distributed=False)))
        # save paper corpus
        self.papers_corpus.papers.coalesce(1).write.csv(os.path.join(self.output_path,Fold.get_papers_corpus_frame_path(self.index, distributed=False)))
        # save lda paper profiles
        # parquet used because we cannot store vectors (lda vectors) in csv format
        self.ldaModel.paper_profiles.coalesce(1).write.parquet(os.path.join(self.output_path,Fold.get_lda_papers_frame_path(self.index, distributed=False)))
        # save candidate set for each paper in the test set
        # format - user_id, [candidate_set]
        # parquet used because we cannot store vectors (lda vectors) in csv format
        self.candidate_set.coalesce(1).write.parquet(os.path.join(self.output_path,Fold.get_candidate_set_data_frame_path(self.index, distributed=False)))
        # save user clusters
        # format - cluster_id, centroid, [user_ids]
        # parquet used because we cannot store vectors (lda vectors) in csv format
        # self.user_clusters.coalesce(1).write.parquet(Fold.get_user_clusters_data_frame_path(self.index, distributed=False))

    @staticmethod
    def get_test_data_frame_path(fold_index, distributed=True):
        """
        Get the path to the file/folder where test data frame for a particular fold was stored. For the identification of a fold
        its fold_index is used. Because a fold can be stored in both distributed(partitioned) and non-distributed(single csv file)
        manner, specify from which you want to load it.
        
        :param fold_index: used for identification of a fold
        :param distributed: if the fold was stored in distributed or non-distributed manner
        :return: path to the file/folder
        """
        if(distributed):
            return Fold.DISTRIBUTED_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.TEST_DF_CSV_FILENAME
        else:
            return Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.TEST_DF_CSV_FILENAME

    @staticmethod
    def get_training_data_frame_path(fold_index, distributed=True):
        """
        Get the path to the file/folder where training data frame for a particular fold was stored. For the identification of a fold
        its fold_index is used. Because a fold can be stored in both distributed(partitioned) and non-distributed(single csv file)
        manner, specify from which you want to load it.

        :param fold_index: used for identification of a fold
        :param distributed: if the fold was stored in distributed or non-distributed manner
        :return: path to the file/folder
        """
        if (distributed):
            return Fold.DISTRIBUTED_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.TRAINING_DF_CSV_FILENAME
        else:
            return Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.TRAINING_DF_CSV_FILENAME

    @staticmethod
    def get_papers_corpus_frame_path(fold_index, distributed=True):
        """
        Get the path to the file/folder where papers corpus data frame for a particular fold was stored. For the identification of a fold
        its fold_index is used. Because a fold can be stored in both distributed(partitioned) and non-distributed(single csv file)
        manner, specify from which you want to load it.

        :param fold_index: used for identification of a fold
        :param distributed: if the fold was stored in distributed or non-distributed manner
        :return: path to the file/folder
        """
        if (distributed):
            return Fold.DISTRIBUTED_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.PAPER_CORPUS_DF_CSV_FILENAME
        else:
            return Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.PAPER_CORPUS_DF_CSV_FILENAME

    @staticmethod
    def get_candidate_set_data_frame_path(fold_index, distributed=True):
        """
        Get the path to the file/folder where candidate set data frame for a particular fold was stored. For the identification of a fold
        its fold_index is used. Because a fold can be stored in both distributed(partitioned) and non-distributed(single csv file)
        manner, specify from which you want to load it.

        :param fold_index: used for identification of a fold
        :param distributed: if the fold was stored in distributed or non-distributed manner
        :return: path to the file/folder
        """
        if (distributed):
            return Fold.DISTRIBUTED_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.CANDIDATE_DF_FILENAME
        else:
            return Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.CANDIDATE_DF_FILENAME

    @staticmethod
    def get_evaluation_results_frame_path(fold_index, model_training, peer_count, pair_generation, distributed=True):
        """
        Get the path to the file/folder where evaluation results for each user were stored. For the identification of a fold
        its fold_index is used. Because a fold can be stored in both distributed(partitioned) and non-distributed(single csv file)
        manner, specify from which you want to load it.

        :param fold_index: used for identification of a fold
        :param distributed: if the fold was stored in distributed or non-distributed manner
        :return: path to the file/folder
        """
        if (distributed):
            return Fold.DISTRIBUTED_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + model_training + "-" + str(peer_count) + "-" + pair_generation + Fold.RESULTS_CSV_FILENAME
        else:
            return Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + model_training + "-" + str(peer_count) + "-" + pair_generation + "-" + Fold.RESULTS_CSV_FILENAME

    @staticmethod
    def get_prediction_data_frame_path(fold_index, model_training, distributed=True):
        """
        Get the path to the file/folder where test data frame for a particular fold was stored. For the identification of a fold
        its fold_index is used. Because a fold can be stored in both distributed(partitioned) and non-distributed(single csv file)
        manner, specify from which you want to load it.

        :param fold_index: used for identification of a fold
        :param distributed: if the fold was stored in distributed or non-distributed manner
        :return: path to the file/folder
        """
        if (distributed):
            return Fold.DISTRIBUTED_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + str(model_training) + "-" + Fold.PREDICTION_DF_FILENAME
        else:
            return Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + str(model_training) + Fold.PREDICTION_DF_FILENAME

    @staticmethod
    def get_lda_papers_frame_path(fold_index, distributed=True):
        """
        Get the path to the file/folder where lda papers data frame for a particular fold was stored. For the identification of a fold
        its fold_index is used. Because a fold can be stored in both distributed(partitioned) and non-distributed(single csv file)
        manner, specify from which you want to load it.

        :param fold_index: used for identification of a fold
        :param distributed: if the fold was stored in distributed or non-distributed manner
        :return: path to the file/folder
        """
        if (distributed):
            return Fold.DISTRIBUTED_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.LDA_DF_FILENAME
        else:
            return Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.LDA_DF_FILENAME

    @staticmethod
    def get_user_clusters_data_frame_path(fold_index, distributed=True):
        """
        Get the path to the file/folder where user cluster data frame for a particular fold was stored. For the identification of a fold
        its fold_index is used. Because a fold can be stored in both distributed(partitioned) and non-distributed(single csv file)
        manner, specify from which you want to load it. User cluster data frame contains cluster_id, centroid, [user_ids] list of user ids
        which belongs to that cluster

        :param fold_index: used for identification of a fold
        :param distributed: if the fold was stored in distributed or non-distributed manner
        :return: path to the file/folder
        """
        if (distributed):
            return Fold.DISTRIBUTED_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.USER_CLUSTERS_DF_FILENAME
        else:
            return Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.USER_CLUSTERS_DF_FILENAME

class FoldCandidateSetGenerator:
    """
    For each user part of the test set generates a number of papers (18 000) that will be its candidate set. A prediction model that is trained
    will be used to predict a score for each of these papers. A candidate set for a user contains 18 000 paper ids (or less for fold 1).
    Random 18 000 papers are selected from [{paper_corpus} - {{test_library} + {training_library}}] where test_library and training_library are papers for a 
    particular user part of its test set and training set, correspondingly. After the candidate set is selected, test library is added to it. This way
    we guarantee that predictions for all papers in the test set will be included in the evaluation phase.
    Format of the candidate set - user_id, candidate_set, where candidate_set is a list of paper ids.
    """

    def __init__(self, spark, paper_corpus, training_data_frame, test_data_frame, userId_col = "user_id", paperId_col = "paper_id", citeulikePaperId_col="citeulike_paper_id"):
        self.spark = spark
        self.training_data_frame = training_data_frame
        self.test_data_frame = test_data_frame
        self.paper_corpus = paper_corpus
        self.userId_col = userId_col
        self.paperId_col = paperId_col
        self.citeulikePaperId_col = citeulikePaperId_col

    def generate_candidate_set(self):

        # take all papers that are part of the corpus
        paper_corpus_list = self.paper_corpus.papers.groupBy().agg(F.collect_list(self.paperId_col).alias("paper_corpus")).collect()[0][0]

        # broadcast all the papers in the corpus
        paper_corpus_list_br = self.spark.sparkContext.broadcast(set(paper_corpus_list))

        def get_candidate_set_per_user(training_library, test_library, k):
            paper_corpus_set = paper_corpus_list_br.value
            training_library_set = set(training_library)
            test_library_set = set(test_library)
            limited_corpus_set = paper_corpus_set - training_library_set - test_library_set
            limited_corpus = list(limited_corpus_set)
            shuffle(limited_corpus)
            candidate_set = limited_corpus[:k] + test_library
            return candidate_set

        get_candidate_set_per_user_udf = F.udf(get_candidate_set_per_user, ArrayType(IntegerType()))

        training_user_library = self.training_data_frame.groupBy(self.userId_col).agg(F.collect_list(self.paperId_col).alias("training_user_library"))
        test_user_library = self.test_data_frame.groupBy(self.userId_col).agg(F.collect_list(self.paperId_col).alias("test_user_library"))


        # add training library to each user
        # user_id|   test_user_library | training_user_library
        test_training_user_library = test_user_library.join(training_user_library, self.userId_col)

        # k adjusts the number of candidate papers that are predicted
        k = 5000
        candidate_set = test_training_user_library.withColumn("candidate_set", get_candidate_set_per_user_udf("training_user_library",
                                                                                           "test_user_library",
                                                                                           F.lit(k)))
        candidate_set = candidate_set.select(self.userId_col, "candidate_set")
        return candidate_set

class FoldUserClustersGenerator:
    """
    TODO write comments
    """

    def __init__(self, spark, lda_paper_profiles, training_data_frame, test_data_frame, k_clusters = 3,
                 userId_col = "user_id", paperId_col = "paper_id", ldaVector_col="lda_vector",
                 clusterId_col ="cluster_id", centroid_col = "centroid"):
        self.spark = spark
        self.lda_paper_profiles = lda_paper_profiles
        self.training_data_frame = training_data_frame
        self.test_data_frame = test_data_frame
        self.k_clusters = k_clusters
        self.userId_col = userId_col
        self.paperId_col = paperId_col
        self.ldaVector_col = ldaVector_col
        self.clusterId_col = clusterId_col
        self.centroid_col = centroid_col

    def generate_clusters(self):
        training_data_frame = self.training_data_frame.select(self.userId_col, self.paperId_col)

        # add lda profiles to each paper
        fold_data_frame = training_data_frame.join(self.lda_paper_profiles, self.paperId_col)
        # group all lda vectors of a user in a list
        user_lda_vectors = fold_data_frame.groupBy(self.userId_col).agg(F.collect_list(self.ldaVector_col).alias("user_lda_vectors"))

        def sum_vectors(vectors):
            vector = vectors[0]
            for i in vectors[1:]:
                vector = vector + i
            return Vectors.dense(vector)
        sum_vectors_udf = F.udf(sum_vectors, VectorUDT())

        # sum vectors into one user profile vector, probability distribution is destroyed
        user_profiles = user_lda_vectors.withColumn("user_profile", sum_vectors_udf("user_lda_vectors")).drop("user_lda_vectors")
        # make predictions using the model over
        user_profiles = MLUtils.convertVectorColumnsToML(user_profiles, "user_profile")

        kmeans = KMeans(k=self.k_clusters, featuresCol="user_profile")
        model = kmeans.fit(user_profiles)

        centroids = model.clusterCenters()
        user_clusters = model.transform(user_profiles).withColumnRenamed("prediction", self.clusterId_col)
        # cluster_id, [user_ids]
        clusters = user_clusters.groupBy(self.clusterId_col).agg(F.collect_list(self.userId_col).alias("user_ids"))
        clusters.persist()

        SSE = self.calculate_SSE(centroids, user_clusters)
        Logger.log("Number of clusters: " + str(self.k_clusters))
        Logger.log("SEE: " + str(SSE))
        return clusters

    def calculate_SSE(self, centroids, user_profiles):
        br_centroids = self.spark.sparkContext.broadcast(centroids)

        def square_eucleadian_distance(cluster_id, x):
            centroid = br_centroids.value[cluster_id]
            distance = numpy.linalg.norm(x - centroid)
            return float(distance * distance)

        square_eucleadian_distance_udf = F.udf(square_eucleadian_distance, DoubleType())
        loc_user_profiles = user_profiles.withColumn("sq_L2", square_eucleadian_distance_udf(self.clusterId_col, "user_profile"))
        sum = loc_user_profiles.groupBy().agg(F.sum('sq_L2').alias('sse')).collect()
        return sum[0][0]

class FoldSplitter:
    """
        Class that contains functionality to split data frame into folds based on its timestamp_col. Each fold consist of training and test data frame.
        When a fold is extracted, it can be stored. So if the folds are stored once, they can be loaded afterwards instead of extracting them again.
    """

    def __init__(self, split_method, output_dir):
        """
        Initialize the splitter
        :param split_method: the split method, options: 'time-aware', 'user-based'
        :param output_dir: the directory where the resulting folds will be stored.
        """
        self.split_method = split_method
        self.output_dir = output_dir
        self.folds_stats = []

    def split_into_folds(self, spark, history, bag_of_words, papers_mapping, timestamp_col="timestamp", period_in_months=6, paperId_col="paper_id",
                         citeulikePaperId_col="citeulike_paper_id", userId_col = "user_id", tf_map_col = "term_occurrence", fold_num = 5):
        """
        :param spark:
        :param history: data frame that will be split. The timestamp_col has to be present. It contains papers' likes of users.
        Each row represents a time when a user likes a paper. The format of the data frame is
        (user_hash, citeulikePaperId_col, timestamp_col, userId_col)
        :param bag_of_words:
        :param papers_mapping: data frame that contains mapping between paper ids and citeulike paper ids.
        :param timestamp_col: the name of the timestamp column by which the splitting is done
        :param period_in_months: number of months that defines the time slot from which rows will be selected for the test
        and training data frame.
        :param paperId_col: name of the column that stores paper ids in the input data frames
        :param citeulikePaperId_col: name of the column that stores citeulike paper ids in the input data frames
        :param userId_col name of the column that stores user ids in the input data frames
        :param tf_map_col name of the tf representation column in bag_of_words data frame. The type of the
        column is Map. It contains key:value pairs where key is the term id and value is #occurence of the term
        in a particular paper
        :param fold_num: The number of folds to be generatd, this parameter is used only if the split_method is not time-aware!
        :return: void
        """
        if self.split_method == 'time-aware':
            self.time_aware_split(spark, history, bag_of_words, papers_mapping, timestamp_col, period_in_months, paperId_col,citeulikePaperId_col, userId_col, tf_map_col)

        if self.split_method == 'user-based':                                  
            self.user_based_split(spark, history, bag_of_words, papers_mapping, paperId_col,citeulikePaperId_col, userId_col, tf_map_col, fold_num)

        # Store statistics for each fold and store it
        Logger.log("Storing statistics for folds.")
        stats_file = os.path.join(self.output_dir,'stats.txt')
        file = open(stats_file, "a")
        stats_header = "fold_index | fold_time |  # UTot | #UTR | #UTS | #dU | #nU | #ITot | #ITR | #ITS | #dI | #nI | #RTot | #RTR | #RTS | #PUTR min/max/avg/std | #PUTS min/max/avg/std | #PITR min/max/avg/std | #PITS min/max/avg/std "
        file.write(stats_header + " \n")
        for stats in self.folds_stats:
            file.write(stats + "\n")
        file.close()

    def user_based_split(self, spark, history, bag_of_words, papers_mapping, paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id", userId_col = "user_id", tf_map_col = "term_occurrence", fold_num = 5):
        """
        :param spark:
        :param history: data frame that will be split. The timestamp_col has to be present. It contains papers' likes of users.
        Each row represents a time when a user likes a paper. The format of the data frame is
        (user_id, citeulike_paper_id, citeulike_user_hash, timestamp, paper_id)
        :param bag_of_words:
        :param papers_mapping: data frame that contains mapping between paper ids and citeulike paper ids.
        :param paperId_col: name of the column that stores paper ids in the input data frames
        :param citeulikePaperId_col: name of the column that stores citeulike paper ids in the input data frames
        :param userId_col name of the column that stores user ids in the input data frames
        :param tf_map_col name of the tf representation column in bag_of_words data frame. The type of the
        :return: void
        """
        Logger.log("Split data into folds using user-based.")

        udfContainer = UDFContainer()
        # add paper id to the ratings
        history = history.join(papers_mapping, citeulikePaperId_col)
        history.cache()

        # Group users ratings
        group_user = history.groupBy('user_id').agg(F.collect_set('paper_id').alias('library'))

        # Randomly split each user library into [fold_num] sets
        df = group_user.withColumn('splits', udfContainer.random_divide(group_user[userId_col],F.lit(fold_num)))

        # Create the folds:
        for fold_index in range(fold_num):
            start_time = datetime.datetime.now()
            col_name = 'Fold_' + str(fold_index + 1)
            test_data_frame = df.select(userId_col, udfContainer.get_test_set_udf('splits', F.lit(fold_index)).alias(col_name))
            test_data_frame = test_data_frame.select(userId_col, F.explode(test_data_frame[col_name]).alias(paperId_col))

            training_data_frame = df.select(userId_col, udfContainer.get_training_set_udf('splits', F.lit(fold_index)).alias(col_name))
            training_data_frame = training_data_frame .select(userId_col, F.explode(training_data_frame [col_name]).alias(paperId_col))

            # construct the fold object
            fold = Fold(training_data_frame, test_data_frame, self.output_dir, self.split_method)
            fold.set_index(fold_index+1)

            # build the corpus for the fold, it includes all papers part of the fold
            fold_papers = fold.training_data_frame.join(papers_mapping,paperId_col).select(citeulikePaperId_col, paperId_col) \
                .union(fold.test_data_frame.join(papers_mapping,paperId_col).select(citeulikePaperId_col, paperId_col)).dropDuplicates()
            fold_papers_corpus = PaperCorpusBuilder.buildCorpus(fold_papers, paperId_col, citeulikePaperId_col)
            fold.set_papers_corpus(fold_papers_corpus)

            # train LDA, topics adjusts the number of topics generated by LDA
            topics = 150
            Logger.log("Training LDA. Fold: {}. The number of topics: {}".format(fold_index, topics))
            ldaVectorizer = LDAVectorizer(papers_corpus=fold_papers_corpus, k_topics=topics, paperId_col=paperId_col, tf_map_col=tf_map_col, output_col="lda_vector")
            ldaModel = ldaVectorizer.fit(bag_of_words)
            fold.ldaModel = ldaModel

            Logger.log("Generate candidate set.")
            # Generate candidate set for each user in the test set
            candidateGenerator = FoldCandidateSetGenerator(spark, fold_papers_corpus, fold.training_data_frame, fold.test_data_frame, userId_col, paperId_col, citeulikePaperId_col)
            candidate_set = candidateGenerator.generate_candidate_set()
            fold.candidate_set = candidate_set

            end_time = datetime.datetime.now() - start_time
            file = open(os.path.join(self.output_dir,"creation-folds.txt"), "a")
            file.write("Split and create fold: " + str(fold_index))
            file.write("End time: " + str(end_time) + "\n")
            file.close()

            Logger.log("Storing of the fold.")
            start_time = datetime.datetime.now()
            fold.store_distributed()
            end_time = datetime.datetime.now() - start_time
            file = open(os.path.join(self.output_dir,"creation-folds.txt"), "a")
            file.write("Store fold: " + str(fold_index))
            file.write("End time: " + str(end_time) + "\n")
            file.close()

            # Compute fold statistics
            self.folds_stats.append(FoldsUtils.compute_fold_statistics(fold,userId_col,paperId_col))
        history.unpersist()




    def time_aware_split(self, spark, history, bag_of_words, papers_mapping, timestamp_col="timestamp", period_in_months=6, paperId_col="paper_id",
                         citeulikePaperId_col="citeulike_paper_id", userId_col = "user_id", tf_map_col = "term_occurrence"):
        """
        Data frame will be split on a timestamp_col based on the period_in_months parameter.
        Initially, by sorting the input data frame by timestamp will be extracted the most recent date and the least
        recent date. Folds are constructed starting for the least recent extracted date. For example, the first fold will
        contain the rows with timestamps in interval [the least recent date, the least recent extracted date + period_in_months]
        as its training set. The test set will contain the rows with timestamps in interval [the least recent date +
        period_in_months , the least recent date + 2 * period_in_months]. For the next fold we include the rows from the next
        "period_in_months" period. And again the rows from the last "period_in_months" period are included in the test set
        and everything else in the training set. Paper corpus contains all papers in a fold.
        Folds information: Currently, in total 5 folds. Data in period [2004-11-04, 2007-12-31].
        1 fold - Training data [2004-11-04, 2005-05-04], Test data [2005-05-04, 2005-11-04]
        2 fold - Training data [2004-11-04, 2005-11-04], Test data [2005-11-04, 2006-05-04]
        3 fold - Training data [2004-11-04, 2006-05-04], Test data [2006-05-04, 2006-11-04]
        4 fold - Training data [2004-11-04, 2006-11-04], Test data [2006-11-04, 2007-05-04]
        5 fold - Training data [2004-11-04, 2007-05-04], Test data [2007-05-04, 2007-11-04]

        :param history: data frame that will be split. The timestamp_col has to be present. It contains papers' likes of users.
        Each row represents a time when a user likes a paper. The format of the data frame is
        (user_hash, citeulikePaperId_col, timestamp_col, userId_col)
        :param papers_mapping: data frame that contains mapping between paper ids and citeulike paper ids.
        :param timestamp_col: the name of the timestamp column by which the splitting is done
        :param period_in_months: number of months that defines the time slot from which rows will be selected for the test
        and training data frame.
        :param paperId_col: name of the column that stores paper ids in the input data frames
        :param citeulikePaperId_col: name of the column that stores citeulike paper ids in the input data frames
        :param userId_col name of the column that stores user ids in the input data frames
        :param tf_map_col name of the tf representation column in bag_of_words data frame. The type of the
        column is Map. It contains key:value pairs where key is the term id and value is #occurence of the term
        in a particular paper
        :return: void
        """
        Logger.log("Split data into folds using time-aware.")
        asc_data_frame = history.orderBy(timestamp_col)
        start_date = asc_data_frame.first()[2]
        Logger.log("Start date:" + str(start_date))

        desc_data_frame = history.orderBy(timestamp_col, ascending=False)
        end_date = desc_data_frame.first()[2]
        Logger.log("End date:" + str(end_date))
        fold_index = 1

        # add paper id to the ratings
        history = history.join(papers_mapping, citeulikePaperId_col)
        history.cache()
        # first fold will contain first "period_in_months" in the training set
        # and next "period_in_months" in the test set
        fold_end_date = start_date + relativedelta(months=2 * period_in_months)
        while fold_end_date < end_date:
            # add the fold to the result list
            # include the next "period_in_months" in the fold, they will be in its test set
            fold_end_date = fold_end_date + relativedelta(months=period_in_months)
            fold_index += 1
            start_time = datetime.datetime.now()
            Logger.log("Extracting fold:" + str(fold_index))
            fold = self.extract_fold(history, fold_end_date, period_in_months, timestamp_col, userId_col)
            # start date of each fold is the least recent date in the input data frame
            fold.set_training_set_start_date(start_date)
            fold.set_index(fold_index)
            # build the corpus for the fold, it includes all papers part of the fold
            fold_papers = fold.training_data_frame.select(citeulikePaperId_col, paperId_col) \
                .union(fold.test_data_frame.select(citeulikePaperId_col, paperId_col)).dropDuplicates()
            fold_papers_corpus = PaperCorpusBuilder.buildCorpus(fold_papers, paperId_col, citeulikePaperId_col)
            fold.set_papers_corpus(fold_papers_corpus)

            # train LDA, topics adjusts the number of topics generated by LDA
            topics = 150
            Logger.log("Training LDA. Fold:" + str(fold_index) + ". The number of topics:" + str(topics))
            ldaVectorizer = LDAVectorizer(papers_corpus=fold_papers_corpus, k_topics=topics,
                                          paperId_col=paperId_col, tf_map_col=tf_map_col,
                                          output_col="lda_vector")
            ldaModel = ldaVectorizer.fit(bag_of_words)
            fold.ldaModel = ldaModel

            Logger.log("Generate candidate set.")
            # Generate candidate set for each user in the test set
            candidateGenerator = FoldCandidateSetGenerator(spark, fold_papers_corpus, fold.training_data_frame,
                                                           fold.test_data_frame,
                                                           userId_col, paperId_col, citeulikePaperId_col)
            candidate_set = candidateGenerator.generate_candidate_set()
            fold.candidate_set = candidate_set

            end_time = datetime.datetime.now() - start_time
            file = open(os.path.join(self.output_dir,"creation-folds.txt"), "a")
            file.write("Split and create fold: " + str(fold_index))
            file.write("End time: " + str(end_time) + "\n")
            file.close()

            Logger.log("Storing of the fold.")
            start_time = datetime.datetime.now()
            fold.store_distributed(self.output_dir)
            end_time = datetime.datetime.now() - start_time
            file = open(os.path.join(self.output_dir,"creation-folds.txt"), "a")
            file.write("Store fold: " + str(fold_index))
            file.write("End time: " + str(end_time) + "\n")
            file.close()

            # add the fold to the result list
            # include the next "period_in_months" in the fold, they will be in its test set
            fold_end_date = fold_end_date + relativedelta(months=period_in_months)
            fold_index += 1
        history.unpersist()

    def extract_fold(self, data_frame, end_date, period_in_months, timestamp_col="timestamp", userId_col = "user_id"):
        """
        Data frame will be split into training and test set based on a timestamp_col and the period_in_months parameter.
        For example, if you have rows with timestamps in interval [2004-11-04, 2011-11-12] in the "data_frame", end_date is 2008-09-29 
        and period_in_months = 6, the test_set will be [2008-09-29, 2009-03-29] and the training_set -> [2004-11-04, 2008-09-28]
        The general rule is rows with timestamp in interval [end_date - period_in_months, end_date] are included in the test set. 
        Rows with timestamp before [end_date - period_in_months] are included in the training set.
        
        :param timestamp_col: the name of the timestamp column by which the splitting is done
        :param userId_col name of the column that stores user ids in the input data frames
        :param data_frame: data frame from which the fold will be extracted. Because we filter based on timestamp, timestamp_col has to be present.
        Its columns: user_hash, citeulikePaperId_col, timestamp_col, userId_col, paper_id
        :param end_date: the end date of the fold that has to be extracted
        :param period_in_months: what time duration will be the test set. Respectively, the training set.
        :return: an object Fold, training and test data frame in the fold have the same format as the input data frame.
        """

        # select only those rows which timestamp is between end_date and (end_date - period_in_months)
        test_set_start_date = end_date + relativedelta(months=-period_in_months)

        # remove all rows outside the period [test_start_date, test_end_date]
        test_data_frame = data_frame.filter(F.col(timestamp_col) >= test_set_start_date).filter(
            F.col(timestamp_col) <= end_date)

        training_data_frame = data_frame.filter(F.col(timestamp_col) < test_set_start_date)

        # all distinct users in training data frame
        user_ids = training_data_frame.select(userId_col).distinct()

        # remove users that are new, part of the test data but not from the training data
        test_data_frame = test_data_frame.join(user_ids, userId_col);

        # construct the fold object
        fold = Fold(training_data_frame, test_data_frame, self.output_dir)
        fold.set_period_in_months(period_in_months)
        fold.set_test_set_start_date(test_set_start_date)
        fold.set_test_set_end_date(end_date)
        return fold

class FoldsUtils:

    @staticmethod
    def store_folds(folds, distributed=True):
        """
        Store folds. Possible both ways - distributed or non-distributed manner.

        :param folds: list of folds that will be stored
        :param distributed: distributed(partitioned) or non-distributed(one single csv file) manner
        """
        for fold in folds:
            if(distributed):
                fold.store_distributed()
            else:
                fold.store()

    @staticmethod
    def compute_fold_statistics(fold, userId_col="user_id", paperId_col="paper_id"):
        """
        Extract statistics from one fold. Each fold consists of test and training data. 
        Statistics that are computed for each fold:
        1) #users in total, TR, TS
        NOTE: new users, remove users in TS that are in TR
        2) #items in total, TR, TS (with or without new items)
        3) #ratings in total, TR, TS
        4) #positive ratings per user - min/max/avg/std in TS/TR
        5) #postive ratings per item - min/max/avg/std in TS/TR
        Possible format of the data frames in the fold - (citeulike_paper_id, paper_id, citeulike_user_hash, user_id).
        :param fold: object that contains information about the fold. It consists of training data frame and test data frame. 
        :param userId_col the name of the column that represents a user by its id or hash
        :param paperId_col the name of the column that represents a paper by its id 
        :return string: statistic formatted in a string.
        """
        Logger.log("Storing statistics for folds.")
        training_data_frame = fold.training_data_frame.select(paperId_col, userId_col)
        test_data_frame = fold.test_data_frame.select(paperId_col, userId_col)
        full_data_set = training_data_frame.union(test_data_frame)

        line = "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | \n"

        # ratings statistics
        total_ratings_count = full_data_set.count()
        tr_ratings_count = training_data_frame.count()
        ts_ratings_count = test_data_frame.count()

        # user statistics #users in total, TR, TS
        total_users_count = full_data_set.select(userId_col).distinct().count()
        tr_users = training_data_frame.select(userId_col).distinct()
        tr_users_count = tr_users.count()
        test_users = test_data_frame.select(userId_col).distinct()
        test_users_count = test_users.count()

        diff_users_count = tr_users.subtract(test_users).count()
        new_users_count = test_users.subtract(tr_users).count()

        # items in total, TR, TS (with or without new items)
        total_items_count = full_data_set.select(paperId_col).distinct().count()
        tr_items = training_data_frame.select(paperId_col).distinct()
        tr_items_count = tr_items.count()
        test_items = test_data_frame.select(paperId_col).distinct()
        test_items_count = test_items.count()

        # papers that appear only in the training set
        diff_items_count = tr_items.subtract(test_items).count()
        # papers that appear only in the test set but not in the training set
        new_items_count = test_items.subtract(tr_items).count()

        # ratings per user - min/max/avg/std in TR
        tr_ratings_per_user = training_data_frame.groupBy(userId_col).agg(F.count("*").alias("papers_count"))
        tr_min_ratings_per_user = tr_ratings_per_user.groupBy().min("papers_count").collect()[0][0]
        tr_max_ratings_per_user = tr_ratings_per_user.groupBy().max("papers_count").collect()[0][0]
        tr_avg_ratings_per_user = tr_ratings_per_user.groupBy().avg("papers_count").collect()[0][0]
        tr_std_ratings_per_user = tr_ratings_per_user.groupBy().agg(F.stddev("papers_count")).collect()[0][0]

        # ratings per user - min/max/avg/std in TS
        ts_ratings_per_user = test_data_frame.groupBy(userId_col).agg(F.count("*").alias("papers_count"))
        ts_min_ratings_per_user = ts_ratings_per_user.groupBy().min("papers_count").collect()[0][0]
        ts_max_ratings_per_user = ts_ratings_per_user.groupBy().max("papers_count").collect()[0][0]
        ts_avg_ratings_per_user = ts_ratings_per_user.groupBy().avg("papers_count").collect()[0][0]
        ts_std_ratings_per_user = ts_ratings_per_user.groupBy().agg(F.stddev("papers_count")).collect()[0][0]

        # ratings per item - min/max/avg/std in TR
        tr_ratings_per_item = training_data_frame.groupBy(paperId_col).agg(F.count("*").alias("ratings_count"))
        tr_min_ratings_per_item = tr_ratings_per_item.groupBy().min("ratings_count").collect()[0][0]
        tr_max_ratings_per_item = tr_ratings_per_item.groupBy().max("ratings_count").collect()[0][0]
        tr_avg_ratings_per_item = tr_ratings_per_item.groupBy().avg("ratings_count").collect()[0][0]
        tr_std_ratings_per_item = tr_ratings_per_item.groupBy().agg(F.stddev("ratings_count")).collect()[0][0]

        # ratings per item - min/max/avg/std in TR
        ts_ratings_per_item = test_data_frame.groupBy(paperId_col).agg(F.count("*").alias("ratings_count"))
        ts_min_ratings_per_item = ts_ratings_per_item.groupBy().min("ratings_count").collect()[0][0]
        ts_max_ratings_per_item = ts_ratings_per_item.groupBy().max("ratings_count").collect()[0][0]
        ts_avg_ratings_per_item = ts_ratings_per_item.groupBy().avg("ratings_count").collect()[0][0]
        ts_std_ratings_per_item = ts_ratings_per_item.groupBy().agg(F.stddev("ratings_count")).collect()[0][0]

        fold_time = str(fold.tr_start_date) + "-" + str(fold.ts_start_date) + "-" + str(fold.ts_end_date)
        formatted_line = line.format(fold.index, fold_time, total_users_count, tr_users_count, test_users_count, diff_users_count, new_users_count, \
                                     total_items_count, tr_items_count, test_items_count, diff_items_count, new_items_count, \
                                     total_ratings_count, tr_ratings_count, ts_ratings_count, \
                                     str(tr_min_ratings_per_user) + "/" + str(tr_max_ratings_per_user) +
                                     "/" + "{0:.2f}".format(tr_avg_ratings_per_user) + "/" + "{0:.2f}".format(tr_std_ratings_per_user) \
                                     , str(ts_min_ratings_per_user) + "/" + str(ts_max_ratings_per_user) + "/" +
                                     "{0:.2f}".format(ts_avg_ratings_per_user) + "/" + "{0:.2f}".format(ts_std_ratings_per_user), \
                                     str(tr_min_ratings_per_item) + "/" + str(tr_max_ratings_per_item) + "/" +
                                     "{0:.2f}".format(tr_avg_ratings_per_item) + "/" + "{0:.2f}".format(tr_std_ratings_per_item), \
                                     str(ts_min_ratings_per_item) + "/" + str(ts_max_ratings_per_item) + "/" +
                                     "{0:.2f}".format(ts_avg_ratings_per_item) + "/" + "{0:.2f}".format(ts_std_ratings_per_item))
        return formatted_line

class FoldValidator():
    """
    Class that run all phases of the Learning To Rank algorithm on multiple folds. It divides the input data set based 
    on a time slot into multiple folds. If they are already stored, before running the algorithm, they will be loaded.
    Otherwise FoldSplitter will be used to extract and store them for later use. Each fold contains test, 
    training data set, papers corpus and LDA model are generated. The model is trained over the training set. 
    Then the prediction over the test set (candidate set) is calculated. 
    """

    """ Total number of folds. Adjust number of folds. """
    NUMBER_OF_FOLD = 5;

    def __init__(self, peer_papers_count=10, pairs_generation="edp", paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id",
                 userId_col="user_id", tf_map_col="term_occurrence", model_training = "gm", output_dir = 'results', split_method = 'time-aware'):
        """
        Construct FoldValidator object.

        :param peer_papers_count: the number of peer papers that will be sampled per paper. See LearningToRank 
        :param pairs_generation: DUPLICATED_PAIRS, ONE_CLASS_PAIRS, EQUALLY_DISTRIBUTED_PAIRS. See Pairs_Generation enum
        :param paperId_col: name of a column that contains identifier of each paper
        :param citeulikePaperId_col: name of a column that contains citeulike identifier of each paper
        :param userId_col: name of a column that contains identifier of each user
        :param tf_map_col: name of the tf representation column in bag_of_words data frame. The type of the 
        column is Map. It contains key:value pairs where key is the term id and value is #occurence of the term
        in a particular paper.
        :param model_training: gm (general model), imp (individual model parallel version), ims (individual model sequential version)
        See Model_Training enum
        :param split_method: string specifies the method to split into folds, available options: time-aware, user-based
        """
        self.paperId_col = paperId_col
        self.userId_col = userId_col
        self.citeulikePaperId_col = citeulikePaperId_col
        self.peer_papers_count = peer_papers_count
        self.pairs_generation = pairs_generation
        self.userId_col = userId_col
        self.tf_map_col = tf_map_col
        self.model_training = model_training
        self.output_dir = output_dir
        self.split_method = split_method

    def create_folds(self, spark, history, bag_of_words, papers_mapping, timestamp_col="timestamp", fold_period_in_months=6):
        """
        Split history data frame into folds based on timestamp_col. For each of them construct its papers corpus using
        all papers of a fold. To extract the folds, FoldSplitter is used. The folds will be stored(see Fold.store()).
        Statistics are also stored for each fold.

        :param history: data frame which contains information when a user liked a paper.
        Its columns: user_hash, citeulikePaperId_col, timestamp_col, userId_col
        :param bag_of_words:(data frame) bag of words representation for each paper. Format (paperId_col, tf_map_col)
        :param papers_mapping: data frame that contains a mapping (citeulikePaperId_col, paperId_col)
        :param statistics_file_name name of the file in which statistics will be written
        :param timestamp_col: the name of the timestamp column by which the splitting is done. It is part of a history data frame
        :param split_method: The method to split data, optinos: 'time-aware', 'user-based'
        :param fold_period_in_months: number of months that defines the time slot from which rows will be selected for the test 
        and training data frame
        """
        # creates all splits and stores them
        Logger.log("Creating folds.")
        start_time = datetime.datetime.now()

        # The output folder of the splitter is a subfolder from the output folder of the validator, the subfolder is named after the split_method:
        splitter = FoldSplitter(self.split_method, os.path.join(self.output_dir,'{}_folds'.format(self.split_method)))
        splitter.split_into_folds(spark, history, bag_of_words, papers_mapping, timestamp_col, fold_period_in_months,self.paperId_col, self.citeulikePaperId_col, self.userId_col)
        end_time = datetime.datetime.now() - start_time
        file = open(os.path.join(self.output_dir,"creation-folds.txt"), "a")
        file.write("Overall time : ")
        file.write("End time: " + str(end_time) + "\n")
        file.close()

    def load_fold(self, spark, fold_index, distributed=True):
        """
        Load a fold based on its index. Loaded fold which contains test data frame, training data frame and papers corpus.
        Structure of test and training data frame - (citeulike_paper_id, citeulike_user_hash, timestamp, user_id, paper_id)
        Structure of papers corpus - (citeulike_paper_id, paper_id)

        :param spark: spark instance used for loading
        :param fold_index: index of a fold that used for identifying its location
        :param distributed if the fold was stored in distributed or non-distributed manner
        :return: loaded fold which contains test data frame, training data frame and papers corpus.
        """
        test_fold_schema = StructType([StructField("user_id", IntegerType(), False),
                                       StructField("citeulike_paper_id", StringType(), False),
                                       StructField("citeulike_user_hash", StringType(), False),
                                       StructField("timestamp", TimestampType(), False),
                                       StructField("paper_id", IntegerType(), False)])
        # load test data frame
        test_data_frame = spark.read.csv(Fold.get_test_data_frame_path(fold_index, distributed), header=False,
            schema=test_fold_schema)

        # load training data frame
        # (name, dataType, nullable)
        training_fold_schema = StructType([StructField("citeulike_paper_id", StringType(), False),
                                           StructField("citeulike_user_hash", StringType(), False),
                                           StructField("timestamp", TimestampType(), False),
                                           StructField("user_id", IntegerType(), False),
                                           StructField("paper_id", IntegerType(), False)])
        training_data_frame = spark.read.csv(Fold.get_training_data_frame_path(fold_index, distributed), header=False,
                                             schema=training_fold_schema)
        fold = Fold(training_data_frame, test_data_frame)
        fold.index = fold_index
        # (name, dataType, nullable)
        paper_corpus_schema = StructType([StructField("citeulike_paper_id", StringType(), False),
                                          StructField("paper_id", IntegerType(), False)])
        # load papers corpus
        papers = spark.read.csv(Fold.get_papers_corpus_frame_path(fold_index, distributed), header=False,
                                schema=paper_corpus_schema)
        fold.papers_corpus = PapersCorpus(papers, paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id")

        # load lda-topics representation for each paper
        papers_lda_vectors = spark.read.parquet(Fold.get_lda_papers_frame_path(fold_index, distributed))
        fold.ldaModel = LDAModel(papers_lda_vectors, paperId_col = "paper_id", output_col = "lda_vector");

        # Load Candidate Set
        candidate_set = spark.read.parquet(Fold.get_candidate_set_data_frame_path(fold_index, distributed))
        fold.candidate_set = candidate_set
        return fold

    def evaluate_folds(self, spark):
        """
        Load each fold, run LTR on it and evaluate its predictions. Then calculate mertics for each fold (MRR@k, RECALL@k, NDCG@k)
        Evaluation are stored for each fold and overall for all folds (avg).
        
        :param spark: spark instance used for loading the folds
        """
        # load folds one by one and evaluate on them
        # total number of fold  - 5
        Logger.log("Start evaluation over folds.")
        for i in range(1, FoldValidator.NUMBER_OF_FOLD + 1):
            # write a file for all folds, it contains a row per fold
            file = open(os.path.join(self.output_dir,"execution.txt"), "a")
            file.write("fold " + str(i) + "\n")
            file.write("Model training: " + self.model_training + "\n")
            file.write("Pair generation: " + self.pairs_generation + "\n")
            file.write("Peer count: " + str(self.peer_papers_count) + "\n")
            file.close()

            Logger.log("Load fold: " + str(i))
            # loading the fold
            start_time = datetime.datetime.now()
            fold = self.load_fold(spark, i)

            # drop some unneeded columns
            # format of test data frame -> user_id | citeulike_paper_id | paper_id |
            fold.test_data_frame = fold.test_data_frame.drop("timestamp", "citeulike_user_hash")
            # format of training data frame -> citeulike_paper_id | user_id | paper_id|
            fold.training_data_frame = fold.training_data_frame.drop("timestamp", "citeulike_user_hash")

            # only for clustered model (CM)
            user_clusters = None
            if(self.model_training == "cmp"):
                userClusterGenerator = FoldUserClustersGenerator(spark, fold.ldaModel.paper_profiles, fold.training_data_frame, fold.test_data_frame, k_clusters = 200,
                     userId_col = "user_id", paperId_col = "paper_id", ldaVector_col="lda_vector", clusterId_col ="cluster_id", centroid_col = "centroid")
                # format - cluster_id, centroid, user_ids
                user_clusters = userClusterGenerator.generate_clusters()
                # format cluster_id, user_id
                user_clusters = user_clusters.withColumn(self.userId_col, F.explode("user_ids")).drop("user_ids")

            # load peers so you can remove the randomization factor when comparing
            # 1) Peer papers sampling
            nps = PeerPapersSampler(fold.papers_corpus, self.peer_papers_count, paperId_col=self.paperId_col,
                                    userId_col=self.userId_col,
                                    output_col="peer_paper_id")

            # generating and saving samples, comment this out to generate new samples
            # peers = nps.transform(fold.training_data_frame)
            # nps.store_peers(i, peers)

            # schema -> user_id | citeulike_paper_id | paper_id | peer_paper_id |
            peers_dataset = nps.load_peers(spark, i)

            # if IMP or IMS, removes from training data frame those users which do not appear in the test set, no need for a model for them to be trained
            if (self.model_training == "imp" or self.model_training == "ims"):
                test_user_ids = fold.test_data_frame.select(self.userId_col).distinct()
                fold.training_data_frame = fold.training_data_frame.join(test_user_ids, self.userId_col)
                peers_dataset = peers_dataset.join(test_user_ids, self.userId_col)

            Logger.log("Persisting the fold ...")
            fold.training_data_frame.persist()
            fold.test_data_frame.persist()
            fold.papers_corpus.papers.persist()
            fold.ldaModel.paper_profiles.persist()
            fold.candidate_set.persist()
            peers_dataset.persist()
            Logger.log("Persist peers...")
            start_time = datetime.datetime.now()

            # Training LTR
            ltr = LearningToRank(spark, fold.papers_corpus, fold.ldaModel, user_clusters=user_clusters, model_training=self.model_training,
                                 pairs_generation=self.pairs_generation, peer_papers_count=self.peer_papers_count,
                                 paperId_col=self.paperId_col, userId_col=self.userId_col, features_col="features")

            Logger.log("Fitting LTR.... .Model:" + str(self.model_training))
            # fit LTR model
            ltr.fit(peers_dataset)

            Logger.log("Making predictions...")

            # predictions by LTR
            Logger.log("Transforming the candidate papers by using the model.")
            candidate_papers_with_predictions = ltr.transform(fold.candidate_set)

            # evaluation
            Logger.log("Starting evaluations...")

            fold_evaluator = FoldEvaluator(k_mrr = [5, 10], k_ndcg = [5, 10], k_recall = [x for x in range(5, 200, 20)],
                                           model_training = self.model_training,
                                           peers_count=self.peer_papers_count,
                                           pairs_generation=self.pairs_generation)
            evaluation_per_user = fold_evaluator.evaluate_fold(candidate_papers_with_predictions, fold, score_col = "ranking_score",  paperId_col = self.paperId_col)

            end_time = datetime.datetime.now() - start_time
            file = open(os.path.join(self.output_dir,"execution.txt"), "a")
            file.write("Overall time:" + str(end_time) + "\n")
            file.close()

            # TODO no need to store results when testing
            # store evaluations per user
            # columns = evaluation_per_user.schema.names
            # fold_evaluator.store_fold_results(fold.index, result, columns, distributed=False)

            # store avg results for a fold
            fold_evaluator.append_fold_overall_result(fold.index, evaluation_per_user, userId_col="user_id")

            Logger.log("Unpersist the data")
            fold.training_data_frame.unpersist()
            fold.test_data_frame.unpersist()
            fold.papers_corpus.papers.unpersist()
            fold.ldaModel.paper_profiles.unpersist()
            peers_dataset.unpersist()
            fold.candidate_set.unpersist()

class FoldEvaluator:
    """ 
    Class that calculates evaluation metrics over a fold. Three metrics are maintained - MRR, NDCG and Recall. 
    """

    """ Name of the file in which results for a fold are written. """
    RESULTS_CSV_FILENAME = "evaluation-results.csv"

    def __init__(self, k_mrr = [5, 10], k_recall = [x for x in range(5, 200, 20)], k_ndcg = [5, 10] , model_training = "gm", peers_count = 1, pairs_generation = "edp"):
        self.k_mrr = k_mrr
        self.k_ndcg = k_ndcg
        self.k_recall = k_recall
        self.model_training = model_training
        self.max_top_k = max((k_mrr + k_ndcg + k_recall))
        self.peers_count = peers_count
        self.pairs_generation = pairs_generation

        column_names = []
        # write a file for all folds, it contains a row per fold
        for k in k_mrr:
            column_names.append("MRR@" + str(k))
        for k in k_recall:
           column_names.append("RECALL@" + str(k))
        for k in k_ndcg:
            column_names.append("NDCG@" + str(k))
        column_names.append("fold_index")
        file_name = os.path.join(self.output_dir,  self.model_training + "-{}-{}-{}".format(self.peers_count, self.pairs_generation, self.RESULTS_CSV_FILENAME))
        with open(file_name, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)


    def evaluate_fold(self, predictions, fold, score_col = "ranking_score", userId_col = "user_id", paperId_col = "paper_id"):
        """
        For each user in the test set, calculate mrr@top_k, recall@top_k and NDCG@top_k. Predictions contain
        for each user predictions for its candidate set of papers.
       
        :param predictions: all candidates for each user with their corresponding predictions. Format (userId_col, paperId_col, score_col)
        :param fold: fold with training and test data sets 
        :param score_col name of a column that contains prediction score for each pair (user, paper)
        :param userId_col name of a column that contains identifier of each user
        :param paperId_col name of a column that contains identifier of each paper
        :return: data frame that contains mrr, recall and ndcg column. They store calculation of these metrics per user
        Format (user_id, mrr, recall, ndcg)
        """

        def mrr_per_user(predicted_papers, test_papers, k):
            """
            Calculate MRR for a specific user. Sort the predicted papers by prediction (DESC order) and select top k.
            Find the first hit in the predicted papers and return 1/(index of the first hit). For
            example, if a test_paper is [7, 12, 19, 66, 10]. And sorted predicted_papers is 
            [(3, 5.5), (4, 4.5) , (5, 4.3), (7, 1.9), (12, 1.5), (10, 1.2), (66, 1.0)]. The first hit 
            is 7 which index is 3. Then the mrr is 1 / (3 + 1)

            :param predicted_papers: list of tuples. Each contains (paper_id, prediction). Not sorted.
            :param test_papers: list of paper ids. Each paper id is part of the test set for a user.
            :param k top # papers needed to be selected
            :return: mrr 
            """
            # checking - if not test set, no evaluation for this user
            if (len(test_papers) == 0):
                return None
            # sort by prediction
            sorted_prediction_papers = sorted(predicted_papers, key=lambda tup: tup[1], reverse=True)
            # take first k
            sorted_prediction_papers = sorted_prediction_papers[:k]
            test_papers_set = set(test_papers)
            index = 1
            for i, prediction in sorted_prediction_papers:
                if (int(i) in test_papers_set):
                    return float(1) / float(index)
                index += 1
            return 0.0

        def recall_per_user(predicted_papers, test_papers, k):
            """
            Calculate Recall for a specific user. Select only the top k paper ids sorted DESC by prediction score.
            Extract only paper ids from predicted_papers, discard prediction information.
            Then, find the number of hits (common paper ids) that both arrays have. Return (#hits)/ (size of test_papers). 
            For example, if a test_paper is [7, 12, 19, 66, 10]. And predicted_papers is  [(3, 5.5), (4, 4.5) , (5, 4.3), (7, 1.9), (12, 1.5), (10, 1.2), (66, 1.0)].
            Hits are [7, 12, 10, 66]. Then the result value is (4 / 5) = 0.8

            :param predicted_papers: list of tuples. Each contains (paper_id, prediction). Not sorted.
            :param test_papers: list of paper ids. Each paper id is part of the test set for a user.
            :param k top # papers needed to be selected
            :return: recall
            """
            # checking - if not test set, no evaluation for this user
            if (len(test_papers) == 0):
                return None
            # sort by prediction
            sorted_prediction_papers = sorted(predicted_papers, key=lambda tup: tup[1], reverse=True)
            # take first k
            sorted_prediction_papers = sorted_prediction_papers[:k]
            predicted_papers = [int(x[0]) for x in sorted_prediction_papers]
            hits = set(predicted_papers).intersection(test_papers)
            if(len(hits) == 0):
                return 0.0
            else:
                return float(len(hits)) / float(len(test_papers))

        def ndcg_per_user(predicted_papers, test_papers, k):
            """
            Calculate NDCG per user. Sort the predicted_papers by their predictions (DESC order) and select top k.
            Then calculate DCG over the predicted papers. DCG is a sum over all predicted papers, for each predicted paper add
            ((prediction) / log(2, position of the paper in the list + 1)). Prediction of a paper is either 0 or 1. If a predicted
            paper is part of the test set, its prediction is 1; otherwise its prediction is 0. Finally ,divide by normalization factor 
            - IDCG calculated over top k predicted papers. IDCG is calculated the same way as DCG, but the only difference is that each 
            paper from top k has prediction/relevance 1.
            For example, if sorted top 5 predicted_papers are [(3, 5.5), (7, 4.5) , (5, 4.3), (6, 4.1), (4, 3.0)] and a test set 
            is [(2, 6.4), (5, 4.5), (3, 1.2)]. DCG will be (1 / log2(2)) + (1 / log2(5)), hits are only paper 3 and paper 5.
            IDCG will be (1 / log2(2)) + (1 / log2(3)) + (1 / log2(4)) + (1 / log2(5)) + (1 / log2(6)))

            :param predicted_papers: list of tuples. Each contains (paper_id, prediction). Not sorted.
            :param test_papers: list of paper ids. Each paper id is part of the test set for a user.
            :param k top # papers needed to be selected
            :return: NDCG for a user
            """
            if (len(test_papers) == 0):
                return None
            # sort by prediction
            sorted_prediction_papers = sorted(predicted_papers, key=lambda tup: tup[1], reverse=True)
            test_papers_set = set(test_papers)
            # take first k
            sorted_prediction_papers = sorted_prediction_papers[:k]
            position = 1
            dcg = 0;
            idcg = 0
            for paper_id, prediction in sorted_prediction_papers:
                # if there is a hit
                if (int(paper_id) in test_papers_set):
                    dcg += 1 / (math.log((position + 1), 2))
                idcg += 1 / (math.log((position + 1), 2))
                position += 1
            return float(dcg) / float(idcg)

        mrr_per_user_udf = F.udf(mrr_per_user, DoubleType())
        ndcg_per_user_udf = F.udf(ndcg_per_user, DoubleType())
        recall_per_user_udf = F.udf(recall_per_user, DoubleType())

        # order by score per user
        window = Window.partitionBy(userId_col).orderBy(F.col(score_col).desc())

        # take max top k size per user
        top_predictions = predictions.select('*', F.rank().over(window).alias('rank')).filter(F.col('rank') <= self.max_top_k)
        # user_id | predictions
        candidate_papers_per_user = top_predictions.groupBy(userId_col).agg(F.collect_list(F.struct(paperId_col, score_col)).alias("predictions"))

        # add test user library to each user
        test_user_library = fold.test_data_frame.groupBy(userId_col).agg(F.collect_list(paperId_col).alias("test_user_library"))

        # user_id | test_user_library | candidate_papers_set |
        evaluation_per_user = test_user_library.join(candidate_papers_per_user, userId_col)

        Logger.log("Adding evaluation...")
        evaluation_columns = []
        # add mrr
        for k in self.k_mrr:
            column_name = "mrr@" + str(k)
            evaluation_columns.append(column_name)
            evaluation_per_user = evaluation_per_user.withColumn(column_name, mrr_per_user_udf("predictions", "test_user_library", F.lit(k)))

        # add recall
        for k in self.k_recall:
            column_name = "recall@" + str(k)
            evaluation_columns.append(column_name)
            evaluation_per_user = evaluation_per_user.withColumn(column_name, recall_per_user_udf("predictions", "test_user_library", F.lit(k)))

        # add ndcg
        for k in self.k_ndcg:
            column_name = "NDCG@" + str(k)
            evaluation_columns.append(column_name)
            evaluation_per_user = evaluation_per_user.withColumn(column_name, ndcg_per_user_udf("predictions", "test_user_library", F.lit(k)))

        # user_id | mrr @ 5 | mrr @ 10 | recall @ 5 | recall @ 25 | recall @ 45 | recall @ 65 | recall @ 85 | recall @ 105 | recall @ 125 | recall @ 145 | recall @ 165 | recall @ 185 | NDCG @ 5 | NDCG @ 10 |
        evaluation_per_user = evaluation_per_user.drop("predictions", "test_user_library")
        Logger.log("Return evaluation per user.")
        return evaluation_per_user

    def store_fold_results(self, fold_index, evaluations_per_user, columns, distributed=True):
        """
        Store evaluation results for a fold.
        
        :param fold_index: identifier of a fold
        :param evaluations_per_user: data frame that contains a row per each user and its evaluation metrics
        :param distributed: 
        :return: if the fold has to be stored in distributed or non-distributed manner
        """
        # save evaluation results
        if(distributed):
            evaluations_per_user.write.csv(Fold.get_evaluation_results_frame_path(fold_index, self.model_training, self.peers_count, self.pairs_generation))
        else:
            filename = Fold.get_evaluation_results_frame_path(fold_index, self.model_training, self.peers_count, self.pairs_generation, distributed=False)
            with open(filename, 'wb') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(columns)
                for i in evaluations_per_user:
                    wr.writerow(i)

    def append_fold_overall_result(self, fold_index, evaluations_per_user, userId_col="user_id"):
        """
        Store avg per user for a fold
        :param overall_evaluation: 
        :return: 
        """

        # drop user id column, we do not need it
        evaluation_per_user = evaluations_per_user.drop(userId_col)
        # take the average over each column
        avg = evaluation_per_user.groupBy().avg()
        overall_evaluation_list = np.array(avg.collect())[0]
        overall_evaluation_list = overall_evaluation_list.tolist()
        overall_evaluation_list.append(fold_index)
        file_name = os.path.join(self.output_dir, self.model_training + "-{}-{}-{}".format(self.peers_count, self.pairs_generation, self.RESULTS_CSV_FILENAME))
        with open(file_name, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(overall_evaluation_list)

        

    
