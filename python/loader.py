from pyspark.sql.types import *

class Loader:
    """
    Class that contains functionality for parsing and loading different files into data frames.
    """

    def __init__(self, path, spark):
        """
        Constructs the loader object.
        
        :param path: path to the folder from where all input files can be read
        :param spark: spark instance needed for loading the data into data frames
        """
        self.path = path
        self.spark = spark

    def load_terms(self, filename):
        """
        Load terms that the paper data set consists of.
        
        :param filename: name of the file that contains the terms. Each line of the file contains one term, id of each term is
        equal to the line number of the term in the file.
        :return: data frame with 2 columns - term, term_id
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
        Loads the file that contains which papers a user likes.  Each line represents a user, the first value is the number of papers in his/her library, 
        the following values are the paper ids separated by a single space. The line number represents the id of the user.
        
        :param filename: name of the file which contains user ratings
        :return: data frame with 3 columns - ratings_count, user_library, user_id
        """
        users_ratings_rdd = self.spark.sparkContext.textFile(self.path + filename)
        users_ratings_rdd = users_ratings_rdd.map(self.__parse_users_ratings)

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

    def __parse_users_ratings(line):
        """
        The method parses a line. Each line represents a user, the first value is the number of papers in 
        his/her library, the following values are the paper ids separated by a single space. 
        
        :return: #ratings, array of paper ids that are part of user's library
        """
        if not line:
            return
        libraryRAW = line.split(' ')
        ratings_count = int(libraryRAW[0])
        # convert to int every paper id
        library = [int(doc_id) for doc_id in libraryRAW[1:]]
        return ratings_count, library

    def load_users_mapping(self, filename):
        """
        Loads the mapping between the user_id (which is used in this data set) and the citeulike_user_hash
        Each line contains citeulike_user_hash and user_id separated by space.
        
        :param filename: name of the file which contains mapping between user_id and citeulike_user_hash
        :return: data frame with two columns - citeulike_user_hash, user_id
        """
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

    def load_papers_mapping(self, filename):
        """
        Loads the mapping between the paper_id (which is used in this data set) and the citeulike_paper_id
        Each line contains citeulike_paper_id and paper_id separated by space.
        
        :param filename: name of the file which contains mapping between paper_id and citeulike_paper_id
        :return: data frame with two columns - paper_id, citeulike_paper_id
        """
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

    def load_papers(self, filename):
        """
        Load papers. They are stored in csv file.  It is comma delimited and has no header.
        
        :param filename: name of the file which contains papers' information. CSV format.
        :return: data frame where each row represents a paper. It consists of 15 columns
        (citeulike_paper_id, type, journal, book_title, series, publisher, pages, 
        volume, number, year, month, postedat, address, title, abstract)
        """
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

    def load_history(self, filename):
        """
        Load the data was downloaded from citeulike. It records for each user (citeulike_user_hash):
        the papers (citeulike_paper_id) he/she added to his/her library along with the timestamp and the tag.
        Loaded file does not have a header and the values are separated by |.

        :param filename: file that stores which papers each user "likes" and "timestamp" when he liked them.
        :return: data frame with 3 columns - citeulike_paper_id, citeulike_user_hash, timestamp
        Note: Because there might be multiple tags per (paper, user) pair which will lead to multiple rows. 
        Duplicates are dropped.
        """
        history = self.spark.sparkContext.textFile(self.path + filename)
        # split each line
        history = history.map(lambda x: x.split("|"))

        # (name, dataType, nullable)
        history_schema = StructType([StructField("citeulike_paper_id", StringType(), False),
                                     StructField("citeulike_user_hash", StringType(), False),
                                     StructField("timestamp", StringType(), False),
                                     StructField("tag", StringType(), False)])
        history = history.toDF(history_schema)
        history = history.drop("tag")
        # # convert timestamp to TimestampType
        history = history.withColumn("timestamp", history.timestamp.cast(TimestampType()))
        # drops duplicates - if there are more tags per (paper, user) pair
        history = history.dropDuplicates()
        return history

    def load_bag_of_words_per_paper(self, filename):
        """
        Load a bag of words for each paper. It links the terms with the papers. Each line has the terms of one paper.
        The first value of each line, is the number of terms appear in the corresponding paper. The following values are space separated key:value pairs. 
        Where the key is the term_id, and the value is the term frequency. The file has no header.

        :param filename: name of the file that contains for each paper a list of term ids that it contains and 
        how many time each term appears in the paper
        :return: data frame with 3 columns - (terms_count, term_occurrence, paper_id)
        """
        bag_of_words_per_paper_rdd = self.spark.sparkContext.textFile(self.path + filename)

        # TODO see what's the problem __parse_bag_of_words not being a static function
        bag_of_words_per_paper_rdd = bag_of_words_per_paper_rdd.map(Loader.__parse_bag_of_words)
        # add 0-based id
        bag_of_words_per_paper_rdd = bag_of_words_per_paper_rdd.zipWithIndex()
        bag_of_words_per_paper_rdd = bag_of_words_per_paper_rdd.map(lambda x: (x[0][0], x[0][1], x[1]))

        # (name, dataType, nullable)
        bag_of_word_schema = StructType([StructField("terms_count", StringType(), False),
                                         StructField("term_occurrence", MapType(StringType(), IntegerType()), False),
                                         StructField("paper_id", IntegerType(), False)])

        bag_of_words_per_paper = bag_of_words_per_paper_rdd.toDF(bag_of_word_schema)
        return bag_of_words_per_paper

    @staticmethod
    def __parse_bag_of_words(line):
        """
        Parse a line. Each line contains the terms of one paper. The first value of each line is the number of terms appeared in the corresponding paper.
        The following values are space separated key:value pairs where the key is the term_id, and the value is the term frequency.
       
        :return: #words, map with key: term_id, value: #occurrence of the key
        """
        if not line:
            return
        bag_of_words_raw = line.split(" ")
        word_count = int(bag_of_words_raw[0])
        bag_of_words_map = [(i.split(":")[0], i.split(":")[1]) for i in bag_of_words_raw[1:]]
        # convert occurrences to int
        terms_occurrences = {term: int(occurrence) for (term, occurrence) in bag_of_words_map}
        return word_count, terms_occurrences
