import os
import pyspark.sql.functions as F
from loader import Loader
from pyspark.sql.types import *
from fold_utils import  FoldValidator
from learning_to_rank import PapersPairBuilder, LearningToRank

# make sure pyspark tells workers to use python3 not 2 if both are installed
os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5'

from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("LTRRecommender2").getOrCreate()
loader = Loader("../citeulike_crawled/terms_keywords_based/", spark);
spark.conf.set("spark.sql.broadcastTimeout", 36000)

def load_reduced_paper_data(file_path):
    papersSchema = StructType([
        #  name, dataType, nullable
        StructField("paper_id", IntegerType(), False),
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
    papers = spark.read.csv(path=file_path, header=False, schema=papersSchema)
    return papers

def load_reduced_history_data(file_path):
    # (name, dataType, nullable)
    history_schema = StructType([StructField("citeulike_paper_id", StringType(), False),
                                 StructField("citeulike_user_hash", StringType(), False),
                                 StructField("timestamp", StringType(), False),
                                 StructField("user_id", IntegerType(), False),
                                 StructField("paper_id", IntegerType(), False)])
    history = spark.read.csv(path=file_path, header=False , schema=history_schema)
    history = history.withColumn("timestamp", history.timestamp.cast(TimestampType()))
    return history

def load_reduced_bag_of_words_data(file_path):
    # (name, dataType, nullable)
    bag_of_words_schema = StructType([StructField("paper_id", StringType(), False),
                                 StructField("terms", StringType(), False)])
    bag_of_words = spark.read.csv(path=file_path, header=False, schema=bag_of_words_schema)
    def parse_terms(line):
        bag_of_words_raw = line.split(" ")
        bag_of_words_map = [(i.split(":")[0], i.split(":")[1]) for i in bag_of_words_raw[1:]]
        # convert occurrences to int
        terms_occurrences = {term: int(occurrence) for (term, occurrence) in bag_of_words_map}
        return terms_occurrences

    parse_terms_udf = F.udf(parse_terms, MapType(StringType(), IntegerType()))
    bag_of_words = bag_of_words.withColumn("term_occurrence", parse_terms_udf("terms"))
    bag_of_words = bag_of_words.drop("terms")
    return bag_of_words

def produce_reduced_set(path):
    papers = loader.load_papers("papers.csv")
    # Loading of the (citeulike paper id - paper id) mapping
    # format (citeulike_paper_id, paper_id)
    papers_mapping = loader.load_papers_mapping("citeulikeId_docId_map.dat")

    papers = papers.join(papers_mapping, "citeulike_paper_id")

    # Loading history
    history = loader.load_history("current")

    # Loading of the (citeulike user hash - user id) mapping
    # format (citeulike_user_hash, user_id)
    user_mappings = loader.load_users_mapping("citeulikeUserHash_userId_map.dat")
    #
    # map citeulike_user_hash to user_id
    # format of history (citeulike_paper_id, citeulike_user_hash, timestamp, tag, paper_id, user_id)
    history = history.join(user_mappings, "citeulike_user_hash", "inner")

    # map citeulike_paper_id to paper_id
    history = history.join(papers_mapping, "citeulike_paper_id", "inner")

    history = history.limit(4000)
    history.write.csv("reduced-data/current.csv")

    # # take all papers of these users
    paper_ids = history.select("paper_id").distinct()

    # # select only papers which paper ids exist in history
    papers = papers.join(paper_ids, "paper_id")
    papers.write.csv("reduced-data/papers.csv")


    # Loading bag of words for each paper
    # format (paper_id, term_occurrences)
    bag_of_words = load_bag_of_words_per_paper("mult.dat")
    bag_of_words = bag_of_words.join(paper_ids, "paper_id")
    bag_of_words.write.csv("reduced-data/mult.csv")


def load_bag_of_words_per_paper(path):
    """
    Load a bag of words for each paper. It links the terms with the papers. Each line has the terms of one paper.
    The first value of each line, is the number of terms appear in the corresponding paper. The following values are space separated key:value pairs. 
    Where the key is the term_id, and the value is the term frequency. The file has no header.

    :param filename: name of the file that contains for each paper a list of term ids that it contains and 
    how many time each term appears in the paper
    :return: data frame with 3 columns - (terms_count, term_occurrence, paper_id)
    """
    bag_of_words_per_paper_rdd = spark.sparkContext.textFile(path)

    # add 0-based id
    bag_of_words_per_paper_rdd = bag_of_words_per_paper_rdd.zipWithIndex()
    bag_of_words_per_paper_rdd = bag_of_words_per_paper_rdd.map(lambda x: (x[0], x[1]))

    # (name, dataType, nullable)
    bag_of_word_schema = StructType([StructField("terms", StringType(), False),
                                     StructField("paper_id", IntegerType(), False)])
    bag_of_words_per_paper = bag_of_words_per_paper_rdd.toDF(bag_of_word_schema)
    return bag_of_words_per_paper

# Loading reduced set of papers
papers = load_reduced_paper_data("reduced-data/papers.csv")

# Loading reduced history data
history = load_reduced_history_data("reduced-data/current.csv")

# paper_id, term_occurrence (map)
bag_of_words = load_reduced_bag_of_words_data("reduced-data/mult.csv")

fold_validator = FoldValidator(bag_of_words, peer_papers_count=10, pairs_generation=PapersPairBuilder.Pairs_Generation.EQUALLY_DISTRIBUTED_PAIRS, paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id",
                  userId_col="user_id", tf_map_col="term_occurrence", model_training=LearningToRank.Model_Training.MODEL_PER_USER)
#fold_validator.create_folds(history, papers, None, "", timestamp_col="timestamp", fold_period_in_months=6)
fold_validator.evaluate_folds(spark)
