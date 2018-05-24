from pyspark.sql import functions as F
from paper_corpus_builder import PaperCorpusBuilder, PapersCorpus
from vectorizers import *
from learning_to_rank_spark2 import *
from pyspark.sql.types import *
from dateutil.relativedelta import relativedelta
from pyspark.sql.window import Window
import datetime

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

    """ Name of the file in which test data frame of the fold is stored. """
    TEST_DF_CSV_FILENAME = "test.csv"
    """ Name of the file in which training data frame of the fold is stored. """
    TRAINING_DF_CSV_FILENAME = "training.csv"
    """ Name of the file in which papers corpus of the fold is stored. """
    PAPER_CORPUS_DF_CSV_FILENAME = "papers-corpus.csv"
    """ Name of the file in which overall results are written. """
    RESULTS_CSV_FILENAME = "results.csv"
    """ Prefix of the name of the folder in which the fold is stored in distributed manner. """
    DISTRIBUTED_PREFIX_FOLD_FOLDER_NAME = "distributed-fold-"
    """ Prefix of the name of the folder in which the fold is stored. """
    PREFIX_FOLD_FOLDER_NAME = "fold-"

    def __init__(self, training_data_frame, test_data_frame):
        self.index = None
        self.training_data_frame = training_data_frame
        self.test_data_frame = test_data_frame
        self.period_in_months = None
        self.tr_start_date = None
        self.ts_end_date = None
        self.ts_start_date = None
        self.papers_corpus = None
        self.folder_path = None

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
        For a fold, store its test data frame, training data frame and its papers corpus.
        All of them are stored in a folder which name is based on PREFIX_FOLD_FOLDER_NAME and the index
        of a fold. For example, for a fold with index 2, the stored information for it will be in
        "distributed-fold-2" folder.
        """
        # save test data frame
        self.test_data_frame.write.csv(Fold.get_test_data_frame_path(self.index))
        # save training data frame
        self.training_data_frame.write.csv(Fold.get_training_data_frame_path(self.index))
        # save paper corpus
        self.papers_corpus.papers.write.csv(Fold.get_papers_corpus_frame_path(self.index))

    def store(self):
        """
        For a fold, store its test data frame, training data frame and its papers corpus.
        Each data frame will be stored in a single csv file.
        All of them are stored in a folder which name is based on PREFIX_FOLD_FOLDER_NAME and the index
        of a fold. For example, for a fold with index 2, the stored information for it will be in
        "fold-2" folder.
        """
        # save test data frame
        self.test_data_frame.coalesce(1).write.csv(Fold.get_test_data_frame_path(self.index, distributed=False))
        # save training data frame
        self.training_data_frame.coalesce(1).write.csv(Fold.get_training_data_frame_path(self.index, distributed=False))
        # save paper corpus
        self.papers_corpus.papers.coalesce(1).write.csv(Fold.get_papers_corpus_frame_path(self.index, distributed=False))

    def persist(self):
        self.test_data_frame.persist()
        self.training_data_frame.persist()
        self.papers_corpus.persist()

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
    def get_evaluation_results_frame_path(fold_index, model_training, distributed=True):
        """
        Get the path to the file/folder where evaluation results for each user were stored. For the identification of a fold
        its fold_index is used. Because a fold can be stored in both distributed(partitioned) and non-distributed(single csv file)
        manner, specify from which you want to load it.

        :param fold_index: used for identification of a fold
        :param distributed: if the fold was stored in distributed or non-distributed manner
        :return: path to the file/folder
        """
        if (distributed):
            return Fold.DISTRIBUTED_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + model_training + "-" + Fold.RESULTS_CSV_FILENAME
        else:
            return Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + model_training + "-" + Fold.RESULTS_CSV_FILENAME

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

class FoldSplitter:
    """
    Class that contains functionality to split data frame into folds based on its timestamp_col. Each fold consist of training and test data frame.
    When a fold is extracted, it can be stored. So if the folds are stored once, they can be loaded afterwards instead of extracting them again.
    """

    def split_into_folds(self, history, papers, papers_mapping, timestamp_col="timestamp", period_in_months=6, paperId_col="paper_id",
                         citeulikePaperId_col="citeulike_paper_id", userId_col = "user_id"):
        """
        Data frame will be split on a timestamp_col based on the period_in_months parameter.
        Initially, by sorting the input data frame by timestamp will be extracted the most recent date and the least 
        recent date. Folds are constructed starting for the least recent extracted date. For example, the first fold will 
        contain the rows with timestamps in interval [the least recent date, the least recent extracted date + period_in_months]
        as its training set. The test set will contain the rows with timestamps in interval [the least recent date + 
        period_in_months , the least recent date + 2 * period_in_months]. For the next fold we include the rows from the next
        "period_in_months" period. And again the rows from the last "period_in_months" period are included in the test set 
        and everything else in the training set. 
        Folds information: Currently, in total 5 folds. Data in period [2004-11-04, 2007-12-31].
        1 fold - Training data [2004-11-04, 2005-05-04], Test data [2005-05-04, 2005-11-04]
        2 fold - Training data [2004-11-04, 2005-11-04], Test data [2005-11-04, 2006-05-04]
        3 fold - Training data [2004-11-04, 2006-05-04], Test data [2006-05-04, 2006-11-04]
        4 fold - Training data [2004-11-04, 2006-11-04], Test data [2006-11-04, 2007-05-04]
        5 fold - Training data [2004-11-04, 2007-05-04], Test data [2007-05-04, 2007-11-04]
        
        :param history: data frame that will be split. The timestamp_col has to be present. It contains papers' likes of users.
        Each row represents a time when a user likes a paper. The format of the data frame is
        (user_id, citeulike_paper_id, citeulike_user_hash, timestamp, paper_id)
        :param papers: data frame that contains representation of all papers. It is used for building a paper corpus for 
        a fold.
        :param papers_mapping: data frame that contains mapping between paper ids and citeulike paper ids. 
        :param timestamp_col: the name of the timestamp column by which the splitting is done
        :param period_in_months: number of months that defines the time slot from which rows will be selected for the test 
        and training data frame.
        :param paperId_col: name of the column that stores paper ids in the input data frames
        :param citeulikePaperId_col: name of the column that stores citeulike paper ids in the input data frames
        :param userId_col name of the column that stores user ids in the input data frames
        :return: list of folds. Each fold is an object Fold. 
        """
        asc_data_frame = history.orderBy(timestamp_col)
        start_date = asc_data_frame.first()[2]

        desc_data_frame = history.orderBy(timestamp_col, ascending=False)
        end_date = desc_data_frame.first()[2]
        fold_index = 1
        folds = []
        # first fold will contain first "period_in_months" in the training set
        # and next "period_in_months" in the test set
        fold_end_date = start_date + relativedelta(months=2 * period_in_months)
        while fold_end_date < end_date:
            fold = FoldSplitter().extract_fold(history, fold_end_date, period_in_months, timestamp_col, userId_col)
            # start date of each fold is the least recent date in the input data frame
            fold.set_training_set_start_date(start_date)
            fold.set_index(fold_index)
            # build the corpus for the fold, it includes all papers published til the end of the fold
            fold_papers_corpus = PaperCorpusBuilder.buildCorpus(papers, papers_mapping, fold_end_date.year , paperId_col, citeulikePaperId_col)

            fold.set_papers_corpus(fold_papers_corpus)
            # add the fold to the result list
            folds.append(fold)
            # include the next "period_in_months" in the fold, they will be in its test set
            fold_end_date = fold_end_date + relativedelta(months=period_in_months)
            fold_index += 1
        return folds

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
        fold = Fold(training_data_frame, test_data_frame)
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
    def write_fold_statistics(folds, statistic_file_name):
        """
        Compute statistics for each fold and store them in a file.
        
        :param folds: list of folds for which statistics will be computed and stored
        :param statistic_file_name: file to which collected statistics are written
        """
        st_writer = FoldStatisticsWriter(statistic_file_name)
        for fold in folds:
            st_writer.statistics(fold)

class FoldValidator():
    """
    Class that run all phases of the Learning To Rank algorithm on multiple folds. It divides the input data set based 
    on a time slot into multiple folds. If they are already stored, before running the algorithm, they will be loaded. Otherwise sFoldSplitter will be used 
    to extract and store them for later use. Each fold contains test, training data set and a papers corpus. The model is trained over the training set. 
    Then the prediction over the test set is calculated. 
    #TODO CONTINUE and fix it
    """

    """ Total number of folds. """
    NUMBER_OF_FOLD = 5;

    # PapersPairBuilder.Pairs_Generation.EQUALLY_DISTRIBUTED_PAIRS
    # LearningToRank.Model_Training.SINGLE_MODEL_ALL_USERS
    def __init__(self, bag_of_words, peer_papers_count=10, pairs_generation="edp", paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id",
                 userId_col="user_id", tf_map_col="term_occurrence", model_training = "sm"):
        """
        Construct FoldValidator object.
        
        :param bag_of_words:(data frame) bag of words representation for each paper. Format (paperId_col, tf_map_col)
        :param peer_papers_count: the number of peer papers that will be sampled per paper. See LearningToRank 
        :param pairs_generation: DUPLICATED_PAIRS, ONE_CLASS_PAIRS, EQUALLY_DISTRIBUTED_PAIRS. See Pairs_Generation enum
        :param paperId_col: name of a column that contains identifier of each paper
        :param citeulikePaperId_col: name of a column that contains citeulike identifier of each paper
        :param userId_col: name of a column that contains identifier of each user
        :param tf_map_col: name of the tf representation column in bag_of_words data frame. The type of the 
        column is Map. It contains key:value pairs where key is the term id and value is #occurence of the term
        in a particular paper.
        :param model_training: MODEL_PER_USER or SINGLE_MODEL_ALL_USERS. See Model_Training enum
        """
        self.paperId_col = paperId_col
        self.userId_col = userId_col
        self.citeulikePaperId_col = citeulikePaperId_col
        self.peer_papers_count = peer_papers_count
        self.bag_of_words = bag_of_words
        self.pairs_generation = pairs_generation
        self.userId_col = userId_col
        self.tf_map_col = tf_map_col
        self.model_training = model_training

    def create_folds(self, history, papers, papers_mapping, statistics_file_name, timestamp_col="timestamp", fold_period_in_months=6):
        """
        Split history data frame into folds based on timestamp_col. For each of them construct its papers corpus using
        papers data frame and papers mapping data frame. Each papers corpus contains all papers published before an end date
        of the fold to which it corresponds. To extract the folds, FoldSplitter is used. The folds will be stored(see Fold.store()).
        Statistics are also stored for each fold.

        :param history: data frame which contains information when a user liked a paper. Its columns timestamp_col, paperId_col,
        citeulikePaperId_col, userId_col
        :param papers: data frame that contains all papers. Its used column is citeulikePaperId_col.
        :param papers_mapping: data frame that contains a mapping (citeulikePaperId_col, paperId_col)
        :param statistics_file_name TODO add
        :param timestamp_col: the name of the timestamp column by which the splitting is done. It is part of a history data frame
        :param fold_period_in_months: number of months that defines the time slot from which rows will be selected for the test 
        and training data frame
        """
        # creates all splits
        folds = FoldSplitter().split_into_folds(history, papers, papers_mapping, timestamp_col, fold_period_in_months,
                                                self.paperId_col,
                                                self.citeulikePaperId_col, self.userId_col)
        # store all folds
        FoldsUtils.store_folds(folds)
        # compute statistics for each fold and store it
        FoldsUtils.write_fold_statistics(folds, statistics_file_name)

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
        return fold

    def load_test_fold(self, spark, fold_index, distributed=True):
        """
        Load a test fold based on its index. Loaded fold which contains test data frame, training data frame and papers corpus.
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
        test_data_frame = spark.read.csv(Fold.LOCAL_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.TEST_DF_CSV_FILENAME, header=False, schema=test_fold_schema)
        # load training data frame
        # (name, dataType, nullable)
        training_fold_schema = StructType([StructField("citeulike_paper_id", StringType(), False),
                                           StructField("citeulike_user_hash", StringType(), False),
                                           StructField("timestamp", TimestampType(), False),
                                           StructField("user_id", IntegerType(), False),
                                           StructField("paper_id", IntegerType(), False)])
        training_data_frame = spark.read.csv(Fold.LOCAL_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.TRAINING_DF_CSV_FILENAME, header=False,
                                             schema=training_fold_schema)
        fold = Fold(training_data_frame, test_data_frame)
        fold.index = fold_index
        # (name, dataType, nullable)
        paper_corpus_schema = StructType([StructField("paper_id", StringType(), False),
                                          StructField("citeulike_paper_id", StringType(), False)])
        # load papers corpus
        papers = spark.read.csv(Fold.LOCAL_PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.PAPER_CORPUS_DF_CSV_FILENAME, header=False,
                                schema=paper_corpus_schema)
        fold.papers_corpus = PapersCorpus(papers, paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id")
        return fold

    def evaluate_folds(self, spark):
        """
        Load each fold, run LTR on it and evaluate its predictions. Then calculate mertics for each fold (MRR@k, RECALL@k, NDCG@k)
        Evaluation are stored for each fold and overall for all folds (avg).
        
        :param spark: spark instance used for loading the folds
        """
        # load folds one by one and evaluate on them
        # total number of fold  - 5
        print("Evaluate folds...")
        for i in range(1, 2): # FoldValidator.NUMBER_OF_FOLD + 1):
            # write a file for all folds, it contains a row per fold
            file = open("results/execution.txt", "a")
            file.write("fold " + str(i) + "\n")
            file.close()

            # loading the fold
            loadingTheFold = datetime.datetime.now()
            fold = self.load_fold(spark, i)

            print("Persisting the fold ...")
            #fold.training_data_frame.persist()
            #fold.test_data_frame.persist()
            #fold.papers_corpus.papers.persist()

            # user_id | citeulike_paper_id | paper_id |
            fold.test_data_frame = fold.test_data_frame.drop("timestamp", "citeulike_user_hash")

            # citeulike_paper_id | user_id | paper_id|
            fold.training_data_frame = fold.training_data_frame.drop("timestamp", "citeulike_user_hash")

            # TODO revert writing in the file
            lf = datetime.datetime.now() - loadingTheFold
            file = open("results/execution.txt", "a")
            file.write("Loading the fold " + str(lf) + "\n")
            file.close()

            print("Start training LDA ...")
            # training TF IDF
            trainingLDA = datetime.datetime.now()
            ldaVectorizer = LDAVectorizer(papers_corpus=fold.papers_corpus, k_topics=5, maxIter=10, paperId_col=self.paperId_col, tf_map_col=self.tf_map_col, output_col="lda_vector")

            print("Paper corpus:")
            print(fold.papers_corpus.papers.count())
            #topics only for testing


            print("LDA trained.")
            tLDA = datetime.datetime.now() - trainingLDA
            print("Fitting LDA...")
            ldaModel = ldaVectorizer.fit(self.bag_of_words)
            file = open("results/execution.txt", "a")
            file.write("Fitting LDA:" + str(tLDA) + "\n")
            file.close()

            # Training LTR
            ltrTraining = datetime.datetime.now()
            ltr = LearningToRank(spark, fold.papers_corpus, ldaModel, pairs_generation=self.pairs_generation, peer_papers_count=self.peer_papers_count,
                                 paperId_col=self.paperId_col, userId_col=self.userId_col, features_col="features", model_training=self.model_training)
            print("Fitting LTR.... (contains LDA fitting).Model:" + str(self.model_training))

            # This is already done when splitting into folds
            # # if PMU or SMMPU, removes from training data frame those users which do not appear in the test set, no need
            # # a model for them to be trained
            # if (self.model_training == "mpu" or self.model_training == "smmu"):
            #     test_user_ids = fold.test_data_frame.select(self.userId_col).distinct()
            #     fold.training_data_frame = fold.training_data_frame.join(test_user_ids, self.userId_col)

            ltr.fit(fold.training_data_frame)

            lrt = datetime.datetime.now() - ltrTraining
            file = open("results/execution.txt", "a")
            file.write("Training LTR(fit)(+ LDA transform), type " + str(self.model_training) + " Time: " + str(lrt) + "\n")
            file.close()
            print("Making predictions...")

            # # PREDICTION by LTR
            ltrPrediction = datetime.datetime.now()
            print("Transforming the paper corpus by using the model")
            papers_corpus_with_predictions = ltr.transform(fold.papers_corpus.papers)

            ltrPr = datetime.datetime.now() - ltrPrediction
            file = open("results/execution.txt", "a")
            file.write("Prediction LTR(transform):" + str(ltrPr) + "\n")
            file.close()
            papers_corpus_with_predictions.collect()

            #print("Persist the predictions.")
            #papers_corpus_with_predictions.persist()

            # paper_id | citeulike_paper_id | ranking_score
            papers_corpus_with_predictions.show()

            # EVALUATION
            eval = datetime.datetime.now()

            # print("Starting evaluations...")
            FoldEvaluator(k_mrr = [5, 10], k_ndcg = [5, 10] , k_recall = [x for x in range(5, 200, 20)], model_training = self.model_training)\
             .evaluate_fold(papers_corpus_with_predictions, fold, score_col = "ranking_score", userId_col = self.userId_col, paperId_col = self.paperId_col)
            evalTime = datetime.datetime.now() - eval
            file = open("results/execution.txt", "a")
            file.write("Evaluation:" + str(evalTime) + "\n")
            file.close()

            print("Unpersist the paper corpus.")
            # fold.papers_corpus.papers.unpersist()
            print("Unpersist the fold and prediction.")
            #papers_corpus_with_predictions.unpersist()
            #fold.training_data_frame.unpersist()
            #fold.test_data_frame.unpersist()


class FoldEvaluator:
    """ 
    Class that calculates evaluation metrics over a fold. Three metrics are maintained - MRR, NDCG and Recall. 
    """

    """ Name of the file in which results for a fold are written. """
    RESULTS_CSV_FILENAME = "evaluation-results.txt"

    def __init__(self, k_mrr = [5, 10], k_recall = [x for x in range(5, 200, 20)], k_ndcg = [5, 10] , model_training = "sm"): #LearningToRank.Model_Training.SINGLE_MODEL_ALL_USERS):
        self.k_mrr = k_mrr
        self.k_ndcg = k_ndcg
        self.k_recall = k_recall
        self.model_training = model_training
        self.max_top_k =  max((k_mrr + k_ndcg + k_recall))

        # write a file for all folds, it contains a row per fold
        file = open( "results/"+ self.RESULTS_CSV_FILENAME, "a")
        header_line = "fold_index |"
        for k in k_mrr:
            header_line = header_line + " MRR@" + str(k) + " |"
        for k in k_recall:
            header_line = header_line + " RECALL@" + str(k) + " |"
        for k in k_ndcg:
            header_line = header_line + " NDCG@" + str(k) + " |"
        # write the header in the file
        file.write(header_line + " \n")
        file.close()


    def evaluate_fold(self, papers_corpus_with_predictions, fold, score_col = "ranking_score", userId_col = "user_id", paperId_col = "paper_id"):
        """
        For each user in the test set, calculate its paper candidate set. It is {paper_corpus}/{training papers} for a user.
        Based on the candidate set and user's test set of papers - calculate mrr@top_k, recall@top_k and NDCG@top_k.
        TODO fix comments
        
        :param papers_corpus_with_predictions: all papers in the corpus with their predictions. Format (paperId_col, "prediction")
        :param fold: fold with training and test data sets 
        :param score_col name of a column that contains prediction score for each pair (user, paper)
        :param userId_col name of a column that contains identifier of each user
        :param paperId_col name of a column that contains identifier of each paper
        :return: data frame that contains mrr, recall and ndcg column. They store calculation of these metrics per user
        Format (user_id, mrr, recall, ndcg)
        """

        def get_candidate_set_per_user(total_predicted_papers, training_papers, k):
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
            sorted_prediction_papers = sorted(total_predicted_papers, key=lambda tup: tup[1], reverse=True)
            training_paper_set = set(training_papers)
            filtered_sorted_prediction_papers = [(float(x), float(y)) for x, y in sorted_prediction_papers if
                                                 x not in training_paper_set]
            return filtered_sorted_prediction_papers[:k]

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
                    return 1 / index
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
                return len(hits) / len(test_papers)

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
            return dcg / idcg

        mrr_per_user_udf = F.udf(mrr_per_user, DoubleType())
        ndcg_per_user_udf = F.udf(ndcg_per_user, DoubleType())
        recall_per_user_udf = F.udf(recall_per_user, DoubleType())
        get_candidate_set_per_user_udf = F.udf(get_candidate_set_per_user, ArrayType(ArrayType(DoubleType())))

        # extract liked papers for each user in the training data set, when top-k for a user is extracted, remove those on which a model is trained
        training_user_library = fold.training_data_frame.groupBy(userId_col).agg(F.collect_list(paperId_col).alias("training_user_library"))
        print("TRaining user library")
        print(training_user_library.count())
        training_user_library_size = training_user_library.select(F.size("training_user_library").alias("tr_library_size"))
        max_training_library = training_user_library_size.groupBy().max("tr_library_size").collect()[0][0]

        candidate_papers_per_user = None
        if(self.model_training == "sm"): #LearningToRank.Model_Training.SINGLE_MODEL_ALL_USERS):
            print("Model training: sm")
            # take max top k + max_training_library size
            papers_corpus_with_predictions = papers_corpus_with_predictions.orderBy(score_col, ascending=False).limit(self.max_top_k + max_training_library)

            top_papers_predictions = papers_corpus_with_predictions.groupBy().agg(F.collect_list(F.struct(paperId_col, score_col)).alias("predictions"))

            # exclude those who do not have test set - This is not valid for MPU and SMMU, because
            # these users were already excluded before the fit/training phase - no model for them
            # was trained. While information about them was used in training SM
            test_user_ids = fold.test_data_frame.select(userId_col).distinct()
            training_user_library = training_user_library.join(test_user_ids, userId_col)

            # add the list of predictions to all selected predicted paper to each user
            candidate_papers_per_user = training_user_library.crossJoin(top_papers_predictions)
        elif(self.model_training == "mpu" or self.model_training == "smmu"): #LearningToRank.Model_Training.MODEL_PER_USER): # SINGLE_MODEL_MULTIPLE_USERS
            print("SMMU evaluation")
            # TODO optimization only users that are part of a test set are needed for evaluation, not all users in the training data set
            print("Before filtering size:")
            print(papers_corpus_with_predictions.count())
            # order by score per user
            window = Window.partitionBy(userId_col).orderBy(F.col(score_col).desc())

            limit = self.max_top_k + max_training_library
            # take max top k + max_training_library size per user
            papers_corpus_with_predictions = papers_corpus_with_predictions.select('*', F.rank().over(window).alias('rank')).filter(F.col('rank') <= limit)

            print("After filtering size:")
            print(papers_corpus_with_predictions.count())
            candidate_papers_per_user = papers_corpus_with_predictions.groupBy(userId_col).agg(F.collect_list(F.struct(paperId_col, score_col)).alias("predictions"))

            print("candidate per user size")
            print(candidate_papers_per_user.count())
            # add training data set to user
            candidate_papers_per_user = candidate_papers_per_user.join(training_user_library, userId_col)

        else:
            # throw an error - unsupported option
            raise ValueError('The option' + self.model_training + ' is not supported.')

        candidate_papers_per_user = candidate_papers_per_user.withColumn("candidate_papers_set", get_candidate_set_per_user_udf("predictions", "training_user_library", F.lit(self.max_top_k)))
        candidate_papers_per_user = candidate_papers_per_user.select(userId_col, "candidate_papers_set")

        print("candiate set:")
        print(candidate_papers_per_user.count())
        candidate_papers_per_user.show()

        # add test user library to each user
        test_user_library = fold.test_data_frame.groupBy(userId_col).agg(F.collect_list(paperId_col).alias("test_user_library"))

        print("test_user_library")
        print(test_user_library.count())
        evaluation_per_user = test_user_library.join(candidate_papers_per_user, userId_col)

        evaluation_per_user.show()
        print(evaluation_per_user.count())

        print("Adding evaluation...")
        evaluation_columns = []
        # add mrr
        for k in self.k_mrr:
            column_name = "mrr@" + str(k)
            evaluation_columns.append(column_name)
            evaluation_per_user = evaluation_per_user.withColumn(column_name, mrr_per_user_udf("candidate_papers_set", "test_user_library", F.lit(k)))

        # add recall
        for k in self.k_recall:
            column_name = "recall@" + str(k)
            evaluation_columns.append(column_name)
            evaluation_per_user = evaluation_per_user.withColumn(column_name, recall_per_user_udf("candidate_papers_set", "test_user_library", F.lit(k)))

        # add ndcg
        for k in self.k_ndcg:
            column_name = "NDCG@" + str(k)
            evaluation_columns.append(column_name)
            evaluation_per_user = evaluation_per_user.withColumn(column_name, ndcg_per_user_udf("candidate_papers_set", "test_user_library", F.lit(k)))

        evaluation_per_user = evaluation_per_user.drop("candidate_papers_set", "test_user_library")
        print("Store evaluation per user.")

        evaluation_per_user.show()
        print("Evaluation count:")
        print(evaluation_per_user.count())
        # store results per fold
        self.store_fold_results(fold.index, self.model_training, evaluation_per_user, distributed=False)

        # store overall results
        # evaluations = {}
        # print("Store overall evaluations...")
        # # compute avg over all columns
        # for metric_column in evaluation_columns:
        #     # remove NoNe values and then calculate the AVG
        #     avg = evaluation_per_user.groupBy().agg(F.avg(metric_column)).collect()
        #     evaluations[metric_column] = avg[0][0]
        # self.append_fold_overall_result(fold.index, evaluations)
        # evaluation_per_user.show()
        return evaluation_per_user

    def store_fold_results(self, fold_index, model_training, evaluations_per_user, distributed=True):
        """
        Store evaluation results for a fold.
        
        :param fold_index: identifier of a fold
        :param evaluations_per_user: data frame that contains a row per each user and its evaluation metrics
        :param distributed: 
        :return: if the fold has to be stored in distributed or non-distributed manner
        """
        # save evaluation results
        if(distributed):
            evaluations_per_user.write.csv(Fold.get_evaluation_results_frame_path(fold_index, model_training))
        else:
            evaluations_per_user.toPandas().to_csv(Fold.get_evaluation_results_frame_path(fold_index, model_training, distributed=False))

    def append_fold_overall_result(self, fold_index, overall_evaluation):
        """
        TODO add comments
        :param overall_evaluation: 
        :return: 
        """
        # write a file for each fold, it contains a row per user
        file = open("results/" + self.RESULTS_CSV_FILENAME, "a")
        line = ""
        line = line + "| " + str(fold_index)
        for metric_name, metric_value in overall_evaluation.items():
            line = line + "| " + str(metric_value)
        file.write(line)
        file.close()

class FoldStatisticsWriter:
    """
    Class that extracts statistics from different folds. For example, number of users in training set, test set and in total.
    """

    def __init__(self, filename):
        """
        Construct an object that is responsible for collecting statistics from folds and store them in a file.
        
        :param filename: the name of the file in which the collected statistics will be written
        """
        self.filename = filename
        file = open("results/" + filename, "a")
        # write the header in the file
        file.write(
            "fold_index | fold_time | #UTot | #UTR | #UTS | #dU | #nU | #ITot | #ITR | #ITS | #dI | #nI |"
            " #RTot | #RTR | #RTS | #PUTR min/max/avg/std | #PUTS min/max/avg/std | #PITR min/max/avg/std | #PITS min/max/avg/std | \n")
        file.close()

    def statistics(self, fold, userId_col="user_id", paperId_col="paper_id"):
        """
        Extract statistics from one fold. Each fold consists of test and training data. 
        Statistics that are computer for each fold:
        1) #users in total, TR, TS
        NOTE: new users, remove users in TS that are in TR
        2) #items in total, TR, TS (with or without new items)
        3) #ratings in total, TR, TS
        4) #positive ratings per user - min/max/avg/std in TS/TR
        5) #postive ratings per item - min/max/avg/std in TS/TR
        
        At the end, write them in a file.
        
        :param fold: object that contains information about the fold. It consists of training data frame and test data frame. 
        :param userId_col the name of the column that represents a user by its id or hash
        :param paperId_col the name of the column that represents a paper by its id 
        Possible format of the data frames in the fold - (citeulike_paper_id, paper_id, citeulike_user_hash, user_id).
        """
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

        # write the collected statistics in a file
        file = open("results/" + self.filename, "a")
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
        file.write(formatted_line)
        file.close()
