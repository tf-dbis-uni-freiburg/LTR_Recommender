from dateutil.relativedelta import relativedelta
import pyspark.sql.functions as F

class FoldSplitter:
    """
    Class that contains functionality to split data frame into folds based on its "timestamp" column. Each fold consist of training and test data frame.
    """

    def split_into_folds(self, data_frame, period_in_months):
        """
        Dataframe will be split on a column "timestamp" based on the period_in_months parameter.
        Initially, by sorting the input data frame by timestamp will be extracted the most recent date and the least recent date. 
        Folds are constructed starting for the least recent extracted date. For example, the first fold will contain the rows
        with timestamps in interval [the least recent date, the least recent extracted date + period_in_months] as its training set. The test set 
        will contain the rows with timestamps in interval [the least recent date + period_in_months , the least recent date + 2 * period_in_months].
        For the next fold we include the rows from the next "period_in_months" period. And again the rows from the last "period_in_months" period are 
        included in the test set and everything else in the training set. 
        Currently, in total 23 folds. Data in period [2004-11-04, 2016-11-11]. Last fold ends in 2016-11-04.
        
        :param data_frame: data frame that will be split. The "timestamp" column has to be present.
        :param period_in_months: number of months that defines the time slot from which rows will be selected for the test and training data frame.
        :return: list of folds. Each fold is an object Fold. 
        """
        asc_data_frame = data_frame.orderBy("timestamp")
        start_date = asc_data_frame.first()[2]
        desc_data_frame = data_frame.orderBy("timestamp", ascending = False)
        end_date = desc_data_frame.first()[2]
        folds = []
        # first fold will contain first "period_in_months" in the training set
        # and next "period_in_months" in the test set
        fold_end_date = start_date + relativedelta(months = 2*period_in_months)
        while fold_end_date < end_date:
            fold = self.extract_fold(data_frame, fold_end_date, period_in_months)
            # start date of each fold is the least recent date in the input data frame
            fold.set_training_set_start_date(start_date)
            folds.append(fold)
            # include the next "period_in_months" in the fold, they will be
            # in its test set
            fold_end_date = fold_end_date + relativedelta(months = period_in_months)
        return folds

    def extract_fold(self, data_frame, end_date, period_in_months):
        """
        Dataframe will be split into training and test set based on a column "timestamp" and the period_in_months parameter.
        For example, if you have rows with timestamps in interval [2004-11-04, 2011-11-12] in the "data_frame", end_date is 2008-09-29 
        and period_in_months = 6, the test_set will be [2008-09-29, 2009-03-29] and the training_set -> [2004-11-04, 2008-09-28]
        The general rule is rows with timestamp in interval [end_date - period_in_months, end_date] are included in the test set. 
        Rows with timestamp before [end_date - period_in_months] are included in the training set.
        
        :param data_frame: data frame from which the fold will be extracted. Because we filter based on timestamp, the "timestamp" column has to be present.
        :param end_date: the end date of the fold that has to be extracted
        :param period_in_months: what time duration will be the test set. Respectively, the training set.
        :return: an object Fold, training and test data drame in the fold have the same format as the input data frame.
        """

        # select only those rows which timestamp is between end_date and (end_date - period_in_months)
        test_set_start_date = end_date + relativedelta(months = -period_in_months)

        # remove all rows outside the period [test_start_date, test_end_date]
        test_data_frame = data_frame.filter(data_frame.timestamp >= test_set_start_date).filter(data_frame.timestamp <= end_date)
        training_data_frame = data_frame.filter(data_frame.timestamp < test_set_start_date)

        # construct the fold object
        fold = Fold(training_data_frame, test_data_frame)
        fold.set_period_in_months(period_in_months)
        fold.set_test_set_end_date(test_set_start_date)
        fold.set_test_set_end_date(end_date)
        return fold

class Fold:
    """
    Encapsulates the notion of a fold. Each fold consists of training and test data frame. 
    Each fold is extracted based on the timestamp of the samples. The samples in the test set are from a particular period in time.
    For example, the test_set can contains samples from [2008-09-29, 2009-03-29] and the training_set from [2004-11-04, 2008-09-28].
    Important note is that the test set starts when the training set ends. The samples are sorted by their timestamp column.
    Therefore, additional information for each fold is when the test set starts and ends. Respectively the same for
    the training set. Also, the duration of the test set - period_in_months.
    """

    def __init__(self, training_data_frame, test_data_frame):
        self.training_data_frame = training_data_frame
        self.test_data_frame = test_data_frame

    def set_period_in_months(self, period_in_months):
       self.period_in_months = period_in_months

    def set_test_set_start_date(self, ts_start_date):
        self.ts_start_date = ts_start_date

    def set_test_set_end_date(self, ts_end_date):
        self.ts_end_date = ts_end_date

    def set_training_set_start_date(self, tr_start_date):
        self.tr_start_date = tr_start_date


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
        file.write("fold_name | #usersTot | #usersTR | #usersTS | #newUsers | #itemsTot | #itemsTR | #itemsTS | #newItems | #ratsTot | #ratsTR | #ratsTS | " +
                   "#PUTR min/max/avg/std | #PUTS min/max/avg/std | #PITR min/max/avg/std | #PITS min/max/avg/std | ")
        file.close()

    def statistics(self, fold):
        #TODO this method depends on "user_id" and "paper_id" columns names !!!
        """
        Extract statistics from one fold. Each fold consists of test and training data.
        At the end, write them in a file.
        :param fold: object that contains information about the fold. It consists of training data frame and test data frame. 
        Format of both of them - (citeulike_paper_id, paper_id, citeulike_user_hash, user_id).
        """

        full_data_set = fold.training_data_frame.union(fold.test_data_frame)

        line = "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |"

        # ratings statistics
        total_ratings_count = full_data_set.count()
        tr_ratings_count = fold.training_data_frame.count()
        ts_ratings_count = fold.test_data_frame.count()

        # user statistics #users in total, TR, TS
        total_users_count = full_data_set.select("user_id").distinct().count()
        tr_users = fold.training_data_frame.select("user_id").distinct()
        tr_users_count = tr_users.count()
        test_users = fold.test_data_frame.select("user_id").distinct()
        test_users_count = test_users.count()
        new_users_count = tr_users.subtract(test_users).count()

        # items in total, TR, TS (with or without new items)
        total_items_count = full_data_set.select("paper_id").distinct().count()
        tr_items = fold.training_data_frame.select("paper_id").distinct()
        tr_items_count = tr_items.count()
        test_items = fold.test_data_frame.select("paper_id").distinct()
        test_items_count = test_items.count()
        new_items_count = tr_items.subtract(test_items).count()

        # positive ratings per user - min/max/avg/std in TR
        tr_ratings_per_user = fold.training_data_frame.groupBy("user_id").agg(F.count("*").alias("papers_count"))
        tr_min_ratings_per_user = tr_ratings_per_user.groupBy().min("papers_count").collect()[0][0]
        tr_max_ratings_per_user = tr_ratings_per_user.groupBy().max("papers_count").collect()[0][0]
        tr_avg_ratings_per_user = tr_ratings_per_user.groupBy().avg("papers_count").collect()[0][0]
        tr_std_ratings_per_user = tr_ratings_per_user.groupBy().agg(F.stddev("papers_count")).collect()[0][0]

        # positive ratings per user - min/max/avg/std in TS
        ts_ratings_per_user = fold.test_data_frame.groupBy("user_id").agg(F.count("*").alias("papers_count"))
        ts_min_ratings_per_user = ts_ratings_per_user.groupBy().min("papers_count").collect()[0][0]
        ts_max_ratings_per_user = ts_ratings_per_user.groupBy().max("papers_count").collect()[0][0]
        ts_avg_ratings_per_user = ts_ratings_per_user.groupBy().avg("papers_count").collect()[0][0]
        ts_std_ratings_per_user = ts_ratings_per_user.groupBy().agg(F.stddev("papers_count")).collect()[0][0]

        # positive ratings per item - min/max/avg/std in TR
        tr_ratings_per_item = fold.training_data_frame.groupBy("paper_id").agg(F.count("*").alias("ratings_count"))
        tr_min_ratings_per_item = tr_ratings_per_item.groupBy().min("ratings_count").collect()[0][0]
        tr_max_ratings_per_item = tr_ratings_per_item.groupBy().max("ratings_count").collect()[0][0]
        tr_avg_ratings_per_item = tr_ratings_per_item.groupBy().avg("ratings_count").collect()[0][0]
        tr_std_ratings_per_item = tr_ratings_per_item.groupBy().agg(F.stddev("ratings_count")).collect()[0][0]

        # positive ratings per item - min/max/avg/std in TR
        ts_ratings_per_item = fold.test_data_frame.groupBy("paper_id").agg(F.count("*").alias("ratings_count"))
        ts_min_ratings_per_item = ts_ratings_per_item.groupBy().min("ratings_count").collect()[0][0]
        ts_max_ratings_per_item = ts_ratings_per_item.groupBy().max("ratings_count").collect()[0][0]
        ts_avg_ratings_per_item = ts_ratings_per_item.groupBy().avg("ratings_count").collect()[0][0]
        ts_std_ratings_per_item = ts_ratings_per_item.groupBy().agg(F.stddev("ratings_count")).collect()[0][0]

        # write the collected statistics in a file
        file = open(self.filename, "w")
        fold_name = fold.tr_start_date + "-" + fold.ts_start_date + "-" + fold.ts_end_date
        formatted_line = line.format(fold_name, total_users_count, tr_users_count, test_users_count, new_users_count, \
                                     total_items_count, tr_items_count, test_items_count, new_items_count, \
                                     total_ratings_count, tr_ratings_count, ts_ratings_count, \
                                     tr_min_ratings_per_user + "/" + tr_max_ratings_per_user+ "/" + tr_avg_ratings_per_user + "/" + tr_std_ratings_per_user, \
                                     ts_min_ratings_per_user + "/" + ts_max_ratings_per_user + "/" + ts_avg_ratings_per_user + "/" + ts_std_ratings_per_user, \
                                     tr_min_ratings_per_item + "/" + tr_max_ratings_per_item + "/" + tr_avg_ratings_per_item + "/" + tr_std_ratings_per_item, \
                                     ts_min_ratings_per_user + "/" + ts_max_ratings_per_user + "/" + ts_avg_ratings_per_user + "/" + ts_std_ratings_per_user)
        file.write(formatted_line)
        file.close()
