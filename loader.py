from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, DateType, MapType, \
    TimestampType

"""
Class that contains functionality for parsing and loading different files into data frames.
"""
class Loader:

    def __init__(self, path, spark):
        """
        :param path: path to the folder from where all input files can be read
        :param spark: spark instance needed for loading the data into dataframes
        """
        self.path = path
        self.spark = spark

    def load_terms(self, filename):
        """
        Load terms that the paper data set consists of.
        
        :param filename: name of the file that contains the terms. Each line of the file contains one term, id of the term is
        equal to the line number of the term in the file.
        :return: dataframe with 2 columns - term and term_id
        """
        terms_rdd = self.spark.sparkContext.textFile(self.path + filename)
        # add index to each term, 0-based on the row
        terms_rdd = terms_rdd.zipWithIndex()
        # (name, dataType, nullable)
        terms_schema = StructType([StructField("term", StringType(), False),
                                   StructField("term_id", IntegerType(), False)])
        # convert to data frame
        terms = terms_rdd.toDF(terms_schema)
        return terms

    def load_user_ratings(self, filename):
        """
        Loads the file that contains which papers a user likes.  Each line contains represent a user, the first value is the number of papers in his/her library, 
        the following values are the paper ids separated by a single space. The line number represents the id of the user.
        :param filename: name of the file which contains user ratings
        :return: dataframe with 3 columns - ratings_count, user_library, user_id
        """
        users_ratings_rdd = self.spark.sparkContext.textFile(self.path + filename)
        users_ratings_rdd = users_ratings_rdd.map(Loader.parse_users_ratings)
        # add index to each term, 0-based on the row
        users_ratings_rdd = users_ratings_rdd.zipWithIndex()
        # extract ratings_counts and array of ratings to 2 different columns
        users_ratings_rdd = users_ratings_rdd.map(lambda x: (x[0][0], x[0][1], x[1]))

        users_ratings_schema = StructType([
            #  name, dataType, nullable
            StructField("ratings_count", IntegerType(), False),
            StructField("user_library", ArrayType(IntegerType(), False), False),
            StructField("user_id", IntegerType(), False)
        ])
        users_ratings = users_ratings_rdd.toDF(users_ratings_schema)
        return users_ratings

    # TODO write comments
    @staticmethod
    def parse_users_ratings(line):
        if not line:
            return
        libraryRAW = line.split(' ')
        ratings_count = int(libraryRAW[0])
        # convert to int every doc id
        library = [int(doc_id) for doc_id in libraryRAW[1:]]
        return ratings_count, library

    # TODO write comments
    def load_users_mapping(self, filename):
        users_mapping = self.spark.sparkContext.textFile(self.path + filename)
        # remove header
        header = users_mapping.first()
        users_mapping = users_mapping.filter(lambda line: line != header)

        # split each line
        users_mapping = users_mapping.map(lambda x: x.split(" "))

        # convert to DF
        users_mapping_schema = StructType([StructField("citeulike_user_hash", StringType(), False),
                                           StructField("user_id", StringType(), False)])
        users_mapping = users_mapping.toDF(users_mapping_schema)
        return users_mapping

    # TODO write comments
    def load_papers_mapping(self, filename):
        papers_mapping = self.spark.sparkContext.textFile(self.path + filename)
        # remove header
        header = papers_mapping.first()
        papers_mapping = papers_mapping.filter(lambda line: line != header)

        # split each line
        papers_mapping = papers_mapping.map(lambda x: x.split(" "))

        # (name, dataType, nullable)
        papers_mapping_schema = StructType([StructField("citeulike_paper_id", StringType(), False),
                                   StructField("paper_id", StringType(), False)])
        papers_mapping = papers_mapping.toDF(papers_mapping_schema)
        return papers_mapping

    # TODO write comments
    def load_papers(self, filename):
        # Load papers into DF
        papersSchema = StructType([
            #  name, dataType, nullable
            StructField("citeulike_paper_id", IntegerType(), False),
            StructField("type", StringType(), True),
            StructField("journal", StringType(), True),
            StructField("book_title", StringType(), True),
            StructField("series", StringType(), True),
            StructField("publisher", StringType(), True),
            StructField("pages", StringType(), True),
            StructField("volume", StringType(), True),
            StructField("number", StringType(), True),
            StructField("year", StringType(), True),
            StructField("month", StringType(), True),
            StructField("postedat", StringType(), True),
            StructField("address", StringType(), True),
            StructField("title", StringType(), True),
            StructField("abstract", StringType(), True)
        ])
        papers = self.spark.read.csv(self.path + filename, header=False, schema=papersSchema)
        return papers

    # TODO write comments
    def load_history(self, filename):
        history = self.spark.sparkContext.textFile(self.path + filename)
        # split each line
        history = history.map(lambda x: x.split("|"))

        # (name, dataType, nullable)
        history_schema = StructType([StructField("citeulike_paper_id", StringType(), False),
                                     StructField("citeulike_user_hash", StringType(), False),
                                     StructField("timestamp", StringType(), False),
                                     StructField("tag", StringType(), False)])
        history = history.toDF(history_schema)
        # drops duplicates - if there are more tags per (paper, user) pair
        history = history.select("citeulike_paper_id", "citeulike_user_hash", "timestamp").dropDuplicates()
        # convert timestamp to TimestampType
        history = history.withColumn("timestamp", history.timestamp.cast(TimestampType()))
        return history

    # TODO write comments
    def load_bag_of_words_per_paper(self, filename):
        bag_of_words_per_paper_rdd = self.spark.sparkContext.textFile(self.path + filename)

        # TODO remove the static function
        bag_of_words_per_paper_rdd = bag_of_words_per_paper_rdd.map(Loader.parse_bag_of_words)
        # add 0-based id
        bag_of_words_per_paper_rdd = bag_of_words_per_paper_rdd.zipWithIndex()
        bag_of_words_per_paper_rdd = bag_of_words_per_paper_rdd.map(lambda x: (x[0][0], x[0][1], x[1]))

        # TODO see if map is the best structure for term occurances
        # (name, dataType, nullable)
        bag_of_word_schema = StructType([StructField("terms_count", StringType(), False),
                                     StructField("term_occurrence", MapType(StringType(), IntegerType()), False),
                                         StructField("paper_id", IntegerType(), False)])

        bag_of_words_per_paper = bag_of_words_per_paper_rdd.toDF(bag_of_word_schema)
        return bag_of_words_per_paper

    # TODO write comments
    @staticmethod
    def parse_bag_of_words(line):
        if not line:
            return
        bag_of_words_raw = line.split(" ")
        word_count = int(bag_of_words_raw[0])
        bag_of_words_map = [(i.split(":")[0], i.split(":")[1]) for i in bag_of_words_raw[1:]]
        # convert occurances to int
        # TODO see if it is not better to convert after it is already loaded in data frame
        terms_occurances = {term : int(occurance) for (term, occurance) in bag_of_words_map}
        return word_count, terms_occurances
