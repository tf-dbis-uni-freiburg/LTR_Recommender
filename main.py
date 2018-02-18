import os
from loader import  Loader
from paper_corpus_builder import PaperCorpusBuilder
from splitter import  FoldSplitter, FoldStatisticsWriter
from vectorizer import *
from negative_papers_sampler import NegativePaperSampler
from papers_pair_builder import PapersPairBuilder
from learning_to_rank import LearningToRank
from spark_utils import *

# make sure pyspark tells workers to use python3 not 2 if both are installed
os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5'

from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("LTRRecommender").getOrCreate()
loader = Loader("../citeulike_crawled/terms_keywords_based/", spark);

papers = loader.load_papers("papers.csv")

# # Loading of the (citeulike paper id - paper id) mapping
# # format (citeulike_paper_id, paper_id)
papers_mapping = loader.load_papers_mapping("citeulikeId_docId_map.dat")

builder = PaperCorpusBuilder()
papers_corpus = builder.buildCorpus(papers, papers_mapping, 2005, paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id")

# Loading history
history = loader.load_history("current")

# Loading of the (citeulike user hash - user id) mapping
# format (citeulike_user_hash, user_id)
user_mappings = loader.load_users_mapping("citeulikeUserHash_userId_map.dat")

# map citeulike_user_hash to user_id
# format of history (citeulike_paper_id, citeulike_user_hash, timestamp, tag, paper_id, user_id)
history = history.join(user_mappings, "citeulike_user_hash", "inner")

# map citeulike_paper_id to paper_id
history = history.join(papers_mapping, "citeulike_paper_id", "inner")

splitter = FoldSplitter()
fold = splitter.extract_fold(history, datetime.datetime(2005, 11, 4), 6, timestamp_col="timestamp")

# Negative papers sampling
nps = NegativePaperSampler(papers_corpus, 10, paperId_col="paper_id", userId_col="user_id", output_col="negative_paper_id")
# generate negative papers - [negative_paper_ids]
training_data_set = nps.transform(fold.training_data_frame)

# Loading bag of words for each paper
# format (paper_id, term_id, term_occurrence)
bag_of_words = loader.load_bag_of_words_per_paper("mult.dat")

# train a model using term_occurrences of each paper and paper corpus
tfVectorizer = TFVectorizer(papers_corpus=papers_corpus, paperId_col="paper_id", tf_map_col="term_occurrence", output_tf_col="positive_paper_tf_vector")
tfVectorizerModel = tfVectorizer.fit(bag_of_words)

# add tf paper representation to each paper based on its paper_id
# add for positive papers
training_data_set = tfVectorizerModel.transform(training_data_set)
# add for negative papers
tfVectorizerModel.setPaperIdCol("negative_paper_id")
tfVectorizerModel.setOutputTfCol("negative_paper_tf_vector")
training_data_set = tfVectorizerModel.transform(training_data_set)
#
# # build pairs
# negative_paper_tf_vector, positive_paper_tf_vector
papersPairBuilder = PapersPairBuilder("equally_distributed_pairs", positive_paperId_col="paper_id", netagive_paperId_col="negative_paper_id",
                 positive_paper_vector_col="positive_paper_tf_vector", negative_paper_vector_col="negative_paper_tf_vector",
                 output_col="pair_paper_difference", label_col="label")
papers_pairs = papersPairBuilder.transform(training_data_set)

# predict using SVM
ltr = LearningToRank(features_col="pair_paper_difference", label_col="label")
lsvcModel = ltr.fit(papers_pairs)
data_set_with_prediction = lsvcModel.transform(papers_pairs)
# (negative_paper_id, positive_paper_id, user_id, citeulike_paper_id, citeulike_user_hash, timestamp, paper_pair_diff, label, rawPrediction, prediction)
data_set_with_prediction.show()

# foldStatistics = FoldStatisticsWriter("statistics.txt")
# sth = foldStatistics.statistics(fold)