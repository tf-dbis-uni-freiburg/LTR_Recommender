import os
import datetime
from loader import  Loader
from paper_corpus_builder import PaperCorpusBuilder
from splitter import  FoldSplitter, FoldStatistics
from tf_vectorizer import  TFVectorizer
from negative_papers_sampler import NegativePaperSampler
from papers_pair_builder import PapersPairBuilder
from learning_to_rank import LearningToRank

# make sure pyspark tells workers to use python3 not 2 if both are installed
os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5'

from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("LTRRecommender").getOrCreate()
loader = Loader("citeulike_crawled/terms_keywords_based/", spark);

papers = loader.load_papers("papers.csv")
builder = PaperCorpusBuilder()
papers_corpus = builder.buildCorpus(papers, 2005)

# Loading of the (citeulike paper id - paper id) mapping
# format (citeulike_paper_id, paper_id)
papers_mappings = loader.load_papers_mapping("citeulikeId_docId_map.dat")
# add paper_id to the corpus
papers_corpus = papers_corpus.join(papers_mappings, "citeulike_paper_id")

# Loading history
history = loader.load_history("current")

# Loading of the (citeulike user hash - user id) mapping
# format (citeulike_user_hash, user_id)
user_mappings = loader.load_users_mapping("citeulikeUserHash_userId_map.dat")

# map citeulike_user_hash to user_id
# format of history (citeulike_paper_id, citeulike_user_hash, timestamp, tag, paper_id, user_id)
history = history.join(user_mappings, "citeulike_user_hash", "inner")

# map citeulike_paper_id to paper_id
history = history.join(papers_mappings, "citeulike_paper_id", "inner")

splitter = FoldSplitter()
training_test_history = splitter.extract_fold(history, datetime.datetime(2005, 9, 28), 6)
training_data_set, test_data_set = training_test_history[0], training_test_history[1]
#
# foldStatistics = FoldStatistics()
# sth = foldStatistics.statistics(training_data_set, test_data_set)

# Negative papers sampling
nps = NegativePaperSampler(spark, papers_corpus, 10)
# generate negative papers - [negative_paper_ids]
training_data_set = nps.transform(training_data_set)

# Loading bag of words for each paper
# format (paper_id, term_id, term_occurrence)
bag_of_words = loader.load_bag_of_words_per_paper("mult.dat")

# train a model using term_occurrences of each paper and paper corpus
tfVectorizer = TFVectorizer(spark, papers_corpus)
tfVectorizerModel = tfVectorizer.fit(bag_of_words)

# add tf paper representation to each paper based on its paper_id
training_data_set = tfVectorizerModel.transform(training_data_set)

# TODO remove this by having an option to set the name of input column
# rename newly generated paper_profile
training_data_set = training_data_set.withColumnRenamed("tf_vector", "positive_paper_tf_vector")
# # rename paper_id
training_data_set = training_data_set.withColumnRenamed("paper_id", "positive_paper_id")
# # rename paper_id
training_data_set = training_data_set.withColumnRenamed("negative_paper_id", "paper_id")
training_data_set = tfVectorizerModel.transform(training_data_set).withColumnRenamed("tf_vector", "negative_paper_tf_vector")
training_data_set = training_data_set.withColumnRenamed("paper_id", "negative_paper_id")

# build pairs
papersPairBuilder = PapersPairBuilder(2)
papers_pairs = papersPairBuilder.transform(training_data_set)

# predict using SVM
ltr = LearningToRank()
# Problem with label -1 ->ERROR LinearSVC: Classification labels should be in [0 to 1]. Found 85240 invalid labels.
lsvcModel = ltr.fit(papers_pairs)
test_data_set_with_prediction = lsvcModel.transform(test_data_set)
test_data_set_with_prediction.show()