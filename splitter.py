from dateutil.relativedelta import relativedelta
import pyspark.sql.functions as F
"""
TODO add class comment +  functions' comments
"""
class FoldSplitter:

    """
    Dataframe will be split on a column "timestamp" based on the period_on_months parameter.
    In total 23 folds. Data in period [2004-11-04, 2016-11-11]. Last fold ends in 2016-11-04.
    """
    def split_into_folds(self, data_frame, period_in_months):
        asc_data_frame = data_frame.orderBy("timestamp")
        start_date = asc_data_frame.first()[2]
        desc_data_frame = data_frame.orderBy("timestamp", ascending=False)
        end_date = desc_data_frame.first()[2]
        # TODO right now we discard the last test_set
        folds = []
        # first fold will contain first "period_in_months" in the training set
        # and next "period_in_months" in the test set
        fold_end_date = start_date + relativedelta(months=2*period_in_months)
        while fold_end_date < end_date:
            fold = self.extract_fold(data_frame, fold_end_date, period_in_months)
            folds.append(fold)
            # include the next "period_in_months" in the fold, they will be
            # in its test set
            fold_end_date = fold_end_date + relativedelta(months=period_in_months)
        return folds

    """
       Dataframe will be split on a column "timestamp" based on the period_on_months parameter.
       Return array which contains [train_data_frame, test_data_frame]
       For example, if you have timestamps in interval [2004-11-04, 2011-11-12], end_date is 2008-09-29 
        and period_in_months = 6, the test_set will be [2008-09-29, 2009-03-29]
        the training_set -> [2004-11-04, 2008-09-28]
       """
    def extract_fold(self, data_frame, end_date, period_in_months):
        # select only those rows which timestamp is between end_date and (end_date - period_in_months)
        test_set_start_date = end_date + relativedelta(months=-period_in_months)

        # remove all rows outside the period [test_start_date, test_end_date]
        test_data_frame = data_frame.filter(data_frame.timestamp >= test_set_start_date).filter(data_frame.timestamp <= end_date).drop("timestamp")
        training_data_frame = data_frame.filter(data_frame.timestamp < test_set_start_date).drop("timestamp")
        return [training_data_frame, test_data_frame]

"""

Class that extracts statistics from different folds. For example, number of users in training set, test set and in total.
"""
class FoldStatistics:


    def statistics(self, training_data_frame, test_data_frame):
        """
        Extract statistics from one fold. Each fold contains of test and training data.
        :param training_data_frame: dataframe that contains training data. Format - (citeulike_paper_id, paper_id, citeulike_user_hash, user_id)
        :param test_data_frame: dataframe that contains test data. Its format is (citeulike_paper_id, paper_id, citeulike_user_hash, user_id)
        :return: 
        """
        full_data_set = training_data_frame.union(test_data_frame)

        # ratings statistics
        total_ratings_count = full_data_set.count()
        print(total_ratings_count)
        tr_ratings_count = training_data_frame.count()
        print(tr_ratings_count)
        ts_ratings_count = test_data_frame.count()
        print(ts_ratings_count)

        # user statistics #users in total, TR, TS
        total_users_count = full_data_set.select("user_id").distinct().count()
        print(total_users_count)
        tr_users = training_data_frame.select("user_id").distinct()
        tr_users_count = tr_users.count()
        print(tr_users_count)
        test_users = test_data_frame.select("user_id").distinct()
        test_users_count = test_users.count()
        print(test_users_count)
        new_users_count = tr_users.subtract(test_users).count()
        print(new_users_count)

        # items in total, TR, TS (with or without new items)
        total_items_count = full_data_set.select("paper_id").distinct().count()
        print(total_items_count)
        tr_items = training_data_frame.select("paper_id").distinct()
        tr_items_count = tr_items.count()
        print(tr_items_count)
        test_items = test_data_frame.select("paper_id").distinct()
        test_items_count = test_items.count()
        print(test_items_count)
        new_items_count = tr_items.subtract(test_items).count()
        print(new_items_count)

        # positive ratings per user - min/max/avg/std in TS/TR
        tr_min_users_count = training_data_frame.groupBy("user_id").agg(F.count("*").alias("papers_count"))
        tr_min_users_count.show()

        # positive ratings per item - min/max/avg/std in TS/TR

