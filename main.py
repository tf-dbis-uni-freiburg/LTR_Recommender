import os
import datetime
from loader import  Loader
from vectorizers import *
from fold_utils import  FoldValidator, FoldSplitter
from paper_corpus_builder import PaperCorpusBuilder

# make sure pyspark tells workers to use python3 not 2 if both are installed
os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5'

from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("LTRRecommender").getOrCreate()
loader = Loader("../citeulike_crawled/terms_keywords_based/", spark);

papers = loader.load_papers("papers.csv")
# # Loading of the (citeulike paper id - paper id) mapping
# # format (citeulike_paper_id, paper_id)
papers_mapping = loader.load_papers_mapping("citeulikeId_docId_map.dat")

papers = papers.join(papers_mapping, "citeulike_paper_id")

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
#
# Loading bag of words for each paper
# format (paper_id, term_occurrences)
bag_of_words = loader.load_bag_of_words_per_paper("mult.dat")

fold_validator = FoldValidator(bag_of_words, peer_papers_count=10, pairs_generation="equally_distributed_pairs", paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id",
                 userId_col="user_id", tf_map_col="term_occurrence")
fold_validator.evaluate(history, papers, papers_mapping, timestamp_col="timestamp", fold_period_in_months=6)

# if folds are already stored and we only load them
# fold_validator.evaluate_folds(spark)

# splitter = FoldSplitter(spark)
# #fold = splitter.extract_fold(history, datetime.datetime(2004, 11, 4), 6, timestamp_col="timestamp")

# # peer papers sampling
# nps = PeerPapersSampler(papers_corpus, 10, paperId_col="paper_id", userId_col="user_id", output_col="peer_paper_id")
# # generate peer papers - [peer_paper_ids]
# training_data_set = nps.transform(fold.training_data_frame)
# test_data_set = nps.transform(fold.test_data_frame)
#
# # train a model using term_occurrences of each paper and paper corpus
# tfVectorizer = TFVectorizer(papers_corpus=papers_corpus, paperId_col="paper_id", tf_map_col="term_occurrence", output_tf_col="paper_tf_vector")
# tfVectorizerModel = tfVectorizer.fit(bag_of_words)
#
# # add tf paper representation to each paper based on its paper_id
# training_data_set = tfVectorizerModel.transform(training_data_set)
# test_data_set = tfVectorizerModel.transform(test_data_set)
#
# # add for peer papers
# tfVectorizerModel.setPaperIdCol("peer_paper_id")
# tfVectorizerModel.setOutputTfCol("peer_paper_tf_vector")
# training_data_set = tfVectorizerModel.transform(training_data_set)
#
# # # build pairs
# # peer_paper_tf_vector, paper_tf_vector
# papersPairBuilder = PapersPairBuilder("equally_distributed_pairs", paperId_col="paper_id", peer_paperId_col="peer_paper_id",
#                  paper_vector_col="paper_tf_vector", peer_paper_vector_col="peer_paper_tf_vector",
#                  output_col="pair_paper_difference", label_col="label")
# training_papers_pairs = papersPairBuilder.transform(training_data_set)
#
# # predict using SVM
# ltr = LearningToRank(features_col="pair_paper_difference", label_col="label")
# lsvcModel = ltr.fit(training_papers_pairs)
#
# test_data_set = test_data_set.withColumnRenamed("paper_tf_vector", "pair_paper_difference")
# test_data_set_with_prediction = lsvcModel.transform(test_data_set)
#
# test_data_set_with_prediction.show()
#
# # metricName=areaUnderROC/areaUnderPR
# evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label",
#                  metricName="areaUnderROC")
# metric = evaluator.evaluate(test_data_set_with_prediction)


