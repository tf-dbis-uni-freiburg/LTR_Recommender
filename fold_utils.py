from dateutil.relativedelta import relativedelta
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from paper_corpus_builder import PaperCorpusBuilder, PapersCorpus
from vectorizers import *
from learning_to_rank import LearningToRank
from pyspark.sql.types import *

class FoldSplitter:
    """
    Class that contains functionality to split data frame into folds based on its timestamp_col. Each fold consist of training and test data frame.
    When a fold is extracted, it can be stored. So if the folds are stored once, they can be loaded afterwards instead of extracting them again.
    """

    @classmethod
    def split_into_folds(history, papers, papers_mapping, timestamp_col="timestamp", period_in_months=6, paperId_col="paper_id",
                         citeulikePaperId_col="citeulike_paper_id", store=True):
        """
        Data frame will be split on a timestamp_col based on the period_in_months parameter.
        Initially, by sorting the input data frame by timestamp will be extracted the most recent date and the least recent date. 
        Folds are constructed starting for the least recent extracted date. For example, the first fold will contain the rows
        with timestamps in interval [the least recent date, the least recent extracted date + period_in_months] as its training set. The test set 
        will contain the rows with timestamps in interval [the least recent date + period_in_months , the least recent date + 2 * period_in_months].
        For the next fold we include the rows from the next "period_in_months" period. And again the rows from the last "period_in_months" period are 
        included in the test set and everything else in the training set. 
        Currently, in total 23 folds. Data in period [2004-11-04, 2016-11-11]. Last fold ends in 2016-11-04.
        
        :param data_frame: data frame that will be split. The timestamp_col has to be present.
        :param timestamp_col: the name of the timestamp column by which the splitting is done
        :param period_in_months: number of months that defines the time slot from which rows will be selected for the test and training data frame.
        :param store: True if the folds have to be stored. See Fold class for more information how the folds are stored.
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
            fold = FoldSplitter.extract_fold(history, fold_end_date, period_in_months)
            # start date of each fold is the least recent date in the input data frame
            fold.set_training_set_start_date(start_date)
            fold.set_index(fold_index)
            # build the corpus for the fold, it includes all papers published til the end of the fold
            fold_papers_corpus = PaperCorpusBuilder.buildCorpus(papers, papers_mapping, fold_end_date.year , paperId_col, citeulikePaperId_col)
            fold.set_papers_corpus(fold_papers_corpus)
            # store the fold
            if(store):
                fold.store()
            # add the fold to the result list
            folds.append(fold)
            # include the next "period_in_months" in the fold, they will be
            # in its test set
            fold_end_date = fold_end_date + relativedelta(months=period_in_months)
            fold_index += 1
        return folds

    @classmethod
    def extract_fold(data_frame, end_date, period_in_months, timestamp_col="timestamp"):
        """
        Data frame will be split into training and test set based on a timestamp_col and the period_in_months parameter.
        For example, if you have rows with timestamps in interval [2004-11-04, 2011-11-12] in the "data_frame", end_date is 2008-09-29 
        and period_in_months = 6, the test_set will be [2008-09-29, 2009-03-29] and the training_set -> [2004-11-04, 2008-09-28]
        The general rule is rows with timestamp in interval [end_date - period_in_months, end_date] are included in the test set. 
        Rows with timestamp before [end_date - period_in_months] are included in the training set.
        
        :param timestamp_col: the name of the timestamp column by which the splitting is done
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

        # construct the fold object
        fold = Fold(training_data_frame, test_data_frame)
        fold.set_period_in_months(period_in_months)
        fold.set_test_set_end_date(test_set_start_date)
        fold.set_test_set_end_date(end_date)
        return fold

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

    def store(self):
        """
        For a fold, store its test data frame, training data frame and its papers corpus.
        All of them are stored in a folder which name is based on PREFIX_FOLD_FOLDER_NAME and the index
        of a fold. For example, for a fold with index 2, the stored information for it will be in
        "fold-2" folder.
        """
        # save test data frame
        self.test_data_frame.write.csv(
            Fold.PREFIX_FOLD_FOLDER_NAME + str(self.index) + "/" + Fold.TEST_DF_CSV_FILENAME)
        # save training data frame
        self.training_data_frame.write.csv(
            Fold.PREFIX_FOLD_FOLDER_NAME + str(self.index) + "/" + Fold.TRAINING_DF_CSV_FILENAME)
        # save paper coprus
        self.paper_corpus.papers.write.csv(
            Fold.PREFIX_FOLD_FOLDER_NAME + str(self.index) + "/" + Fold.PAPER_CORPUS_DF_CSV_FILENAME)

class FoldValidator():
    """
    Class that run all phases of the Learning To Rank algorithm on multiple folds. It divides the input data set based 
    on a time slot into multiple folds. If they are already stored, before running the algorithm, they will be loaded. Otherwise sFoldSplitter will be used 
    to extract and store them for later use. Each fold contains test, training data set and a papers corpus. The model is trained over the training set. 
    Then the prediction over the test set is calculated. 
    #TODO CONTINUE
    """

    """ Total number of folds. """
    NUMBER_OF_FOLD = 23;

    def __init__(self, bag_of_words, k=10,
                 pairs_generation="equally_distributed_pairs", paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id",
                 userId_col="user_id", tf_map_col="term_occurrence"):
        """
        Construct FoldValidator object.
        
        :param bag_of_words:(dataframe) bag of words representation for each paper. Format (paperId_col, tf_map_col)
        :param k: the number of peer papers that will be sampled per paper. See LearningToRank 
        :param pairs_generation: duplicated_pairs, one_class_pairs, equally_distributed_pairs. See LearningToRank
        :param paperId_col: name of a column that contains identifier of each paper
        :param citeulikePaperId_col: name of a column that contains citeulike identifier of each paper
        :param userId_col: name of a column that contains identifier of each user
        :param tf_map_col: name of the tf representation column in bag_of_words data frame. The type of the 
        column is Map. It contains key:value pairs where key is the term id and value is #occurence of the term
        in a particular paper.
        """
        self.paperId_col = paperId_col
        self.citeulikePaperId_col = citeulikePaperId_col
        self.k = k
        self.bag_of_words = bag_of_words
        self.pairs_generation = pairs_generation
        self.userId_col = userId_col
        self.tf_map_col = tf_map_col
        self.folds_evaluation = {}

    def evaluate_folds(self, spark):
        """
        Load each fold, run LTR on it and evaluate its predictions.
        # TODO continue it when the method is finished.
        
        :param spark: spark instance used for loading the folds
        """
        # load folds one by one and evaluate on them
        # total number of fold  - 23
        for i in range(1, FoldValidator.NUMBER_OF_FOLD):
            fold = self.load_fold(spark, i)
            # train a model using term_occurrences of each paper and paper corpus
            tfidfVectorizer = TFIDFVectorizer(papers_corpus=fold.papers_corpus, paperId_col=self.paperId_col,
                                              tf_map_col=self.tf_map_col, output_col="paper_tf_idf_vector")
            tfidfModel = tfidfVectorizer.fit(self.bag_of_words)

            ltr = LearningToRank(fold.papers_corpus, tfidfModel, pairs_generation="equally_distributed_pairs", k=10,
                                 paperId_col="paper_id",
                                 userId_col="user_id", features_col="features")
            training_data_frame = fold.training_data_frame
            lsvcModel = ltr.fit(training_data_frame)

            training_data_frame.show()
            # predict for each fold
            # TODO CONTINUE


    def evaluate(self, history, papers, papers_mapping, timestamp_col="timestamp", fold_period_in_months=6):
        """
        Split history data frame into folds based on timestamp_col. For each of them construct its papers corpus using
        papers data frame and papers mapping data frame. Each papers corpus contains all papers published before an end date
        of the fold to which it corresponds. To extract the folds, FoldSplitter is used. The folds will be stored(see Fold.store()).
        A LTR algorithm is run over each fold.
        # TODO continue it when the method is finished.
        
        :param history: data frame which contains information when a user liked a paper. Its columns timestamp_col, paperId_col,
        citeulikePaperId_col, userId_col
        :param papers: data frame that contains all papers. Its used column is citeulikePaperId_col.
        :param papers_mapping: data frame that contains a mapping (citeulikePaperId_col, paperId_col)
        :param timestamp_col: the name of the timestamp column by which the splitting is done. It is part of a history data frame
        :param fold_period_in_months: number of months that defines the time slot from which rows will be selected for the test and training data frame
        """
        # creates all splits
        folds = FoldSplitter.split_into_folds(history, papers, papers_mapping, timestamp_col, fold_period_in_months,
                                                paperId_col = self.paperId_col, citeulikePaperId_col = self.citeulikePaperId_col, store=True)
        for fold in folds:
            # train a model using term_occurrences of each paper and paper corpus
            tfidfVectorizer = TFIDFVectorizer(papers_corpus=fold.papers_corpus, paperId_col=self.paperId_Col,
                                              tf_map_col=self.tf_map_col, output_col="paper_tf_idf_vector")
            tfidfModel = tfidfVectorizer.fit(self.bag_of_words)

            ltr = LearningToRank(fold.papers_corpus, tfidfModel, pairs_generation="equally_distributed_pairs", k=10,
                                 paperId_col="paper_id",
                                 userId_col="user_id", features_col="features")
            training_data_frame = fold.training_data_frame
            lsvcModel = ltr.fit(training_data_frame)

            training_data_frame.show()
            # predict for each fold
            # TODO CONTINUE

    def load_fold(self, spark, fold_index):
        """
        Load a fold based on its index. Loaded fold which contains test data frame, training data frame and papers corpus.
        Structure of test and training data frame - (citeulike_paper_id, citeulike_user_hash, timestamp, user_id, paper_id)
        Structure of papers corpus - (citeulike_paper_id, paper_id)
        
        :param spark: spark instance used for loading
        :param fold_index: index of a fold that used for identifying its location
        :return: loaded fold which contains test data frame, training data frame and papers corpus.
        """
        # (name, dataType, nullable)
        fold_schema = StructType([StructField("citeulike_paper_id", StringType(), False),
                                  StructField("citeulike_user_hash", StringType(), False),
                                  StructField("timestamp", TimestampType(), False),
                                  StructField("user_id", IntegerType(), False),
                                  StructField("paper_id", IntegerType(), False)])
        # load test data frame
        test_data_frame = spark.read.csv(Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.TEST_DF_CSV_FILENAME, header=False,
                                              schema=fold_schema)
        # load training data frame
        training_data_frame = spark.read.csv(Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.TRAINING_DF_CSV_FILENAME,
                                                  header=False, schema=fold_schema)

        fold = Fold(training_data_frame, test_data_frame)
        fold.index = fold_index
        # (name, dataType, nullable)
        paper_corpus_schema = StructType([StructField("citeulike_paper_id", StringType(), False), StructField("paper_id", IntegerType(), False)])
        # load papers corpus
        papers = spark.read.csv(Fold.PREFIX_FOLD_FOLDER_NAME + str(fold_index) + "/" + Fold.PAPER_CORPUS_DF_CSV_FILENAME,
            header=False, schema=paper_corpus_schema)
        fold.papers_corpus = PapersCorpus(papers, paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id")
        return fold


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
        file = open(filename, "w")
        # write the header in the file
        file.write(
            "fold_name | #usersTot | #usersTR | #usersTS | #newUsers | #itemsTot | #itemsTR | #itemsTS | #newItems | #ratsTot | #ratsTR | #ratsTS | " +
            "#PUTR min/max/avg/std | #PUTS min/max/avg/std | #PITR min/max/avg/std | #PITS min/max/avg/std | ")
        file.close()

    def statistics(self, fold, userId_col="user_id", paperId_col="paper_id"):
        """
        Extract statistics from one fold. Each fold consists of test and training data. TODO write what kind of statistics are computed.
        At the end, write them in a file.
        
        :param fold: object that contains information about the fold. It consists of training data frame and test data frame. 
        :param userId_col the name of the column that represents a user by its id or hash
        :param paperId_col the name of the column that represents a paper by its id 
        Possible format of the data frames in the fold - (citeulike_paper_id, paper_id, citeulike_user_hash, user_id).
        """
        training_data_frame = fold.training_data_frame.select(paperId_col, userId_col)
        test_data_frame = fold.test_data_frame.select(paperId_col, userId_col)
        fold_name = str(fold.tr_start_date) + "-" + str(fold.ts_start_date) + "-" + str(fold.ts_end_date)
        full_data_set = training_data_frame.union(test_data_frame)

        line = "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |"

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
        new_users_count = tr_users.subtract(test_users).count()

        # items in total, TR, TS (with or without new items)
        total_items_count = full_data_set.select(paperId_col).distinct().count()
        tr_items = training_data_frame.select(paperId_col).distinct()
        tr_items_count = tr_items.count()
        test_items = test_data_frame.select(paperId_col).distinct()
        test_items_count = test_items.count()
        new_items_count = tr_items.subtract(test_items).count()

        # ratings per user - min/max/avg/std in TR
        # tr_ratings_per_user = training_data_frame.groupBy(userId_col).agg(F.count("*").alias("papers_count"))
        # tr_min_ratings_per_user = tr_ratings_per_user.groupBy().min("papers_count").collect()[0][0]
        # tr_max_ratings_per_user = tr_ratings_per_user.groupBy().max("papers_count").collect()[0][0]
        # tr_avg_ratings_per_user = tr_ratings_per_user.groupBy().avg("papers_count").collect()[0][0]
        # tr_std_ratings_per_user = tr_ratings_per_user.groupBy().agg(F.stddev("papers_count")).collect()[0][0]
        #
        # # ratings per user - min/max/avg/std in TS
        # ts_ratings_per_user = test_data_frame.groupBy(userId_col).agg(F.count("*").alias("papers_count"))
        # ts_min_ratings_per_user = ts_ratings_per_user.groupBy().min("papers_count").collect()[0][0]
        # ts_max_ratings_per_user = ts_ratings_per_user.groupBy().max("papers_count").collect()[0][0]
        # ts_avg_ratings_per_user = ts_ratings_per_user.groupBy().avg("papers_count").collect()[0][0]
        # ts_std_ratings_per_user = ts_ratings_per_user.groupBy().agg(F.stddev("papers_count")).collect()[0][0]
        #
        # # ratings per item - min/max/avg/std in TR
        # tr_ratings_per_item = training_data_frame.groupBy(paperId_col).agg(F.count("*").alias("ratings_count"))
        # tr_min_ratings_per_item = tr_ratings_per_item.groupBy().min("ratings_count").collect()[0][0]
        # tr_max_ratings_per_item = tr_ratings_per_item.groupBy().max("ratings_count").collect()[0][0]
        # tr_avg_ratings_per_item = tr_ratings_per_item.groupBy().avg("ratings_count").collect()[0][0]
        # tr_std_ratings_per_item = tr_ratings_per_item.groupBy().agg(F.stddev("ratings_count")).collect()[0][0]
        #
        # # ratings per item - min/max/avg/std in TR
        # ts_ratings_per_item = test_data_frame.groupBy(paperId_col).agg(F.count("*").alias("ratings_count"))
        # ts_min_ratings_per_item = ts_ratings_per_item.groupBy().min("ratings_count").collect()[0][0]
        # ts_max_ratings_per_item = ts_ratings_per_item.groupBy().max("ratings_count").collect()[0][0]
        # ts_avg_ratings_per_item = ts_ratings_per_item.groupBy().avg("ratings_count").collect()[0][0]
        # ts_std_ratings_per_item = ts_ratings_per_item.groupBy().agg(F.stddev("ratings_count")).collect()[0][0]

        # write the collected statistics in a file
        file = open(self.filename, "w")
        formatted_line = line.format(fold_name, total_users_count, tr_users_count, test_users_count, new_users_count, \
                                     total_items_count, tr_items_count, test_items_count, new_items_count, \
                                     total_ratings_count, tr_ratings_count, ts_ratings_count)  # , \
        # tr_min_ratings_per_user + "/" + tr_max_ratings_per_user+ "/" + tr_avg_ratings_per_user + "/" + tr_std_ratings_per_user, \
        # ts_min_ratings_per_user + "/" + ts_max_ratings_per_user + "/" + ts_avg_ratings_per_user + "/" + ts_std_ratings_per_user, \
        # tr_min_ratings_per_item + "/" + tr_max_ratings_per_item + "/" + tr_avg_ratings_per_item + "/" + tr_std_ratings_per_item, \
        # ts_min_ratings_per_item + "/" + ts_max_ratings_per_item + "/" + ts_avg_ratings_per_item + "/" + ts_std_ratings_per_item)
        file.write(formatted_line)
        file.close()
