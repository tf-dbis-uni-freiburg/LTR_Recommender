import argparse
import os
from pyspark.sql import SparkSession
from fold_utils import FoldValidator
from loader import Loader

# make sure pyspark tells workers to use python2.7 instead of 3 if both are installed
# uncomment only during development
# os.environ['PYSPARK_PYTHON'] = '/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/System/Library/Frameworks/Python.framework/Versions/2/7/bin/python2.7'

# parse input parameters
parser = argparse.ArgumentParser(description='Process parameters needed to run the program.')
parser.add_argument('-input', type=str, required=True,
                    help='folder from where input files are read')
parser.add_argument('-peers_count', type=int, default=2,
                    help='number of peer papers generated for a paper')
parser.add_argument('-pairs_generation', type=str, default="edp",
                    help='Approaches for generating pairs. Possible options: 1) duplicated_pairs - dp , 2) one_class_pairs - ocp, 3) equally_distributed_pairs - edp')
parser.add_argument('-model_training', type=str, default="smmu",
                    help='Different training approaches for LTR. Possible options 1) model per user - mpu 2) single model for all users - sm 3) one model that contains different weight vectors for each user - smmu')
args = parser.parse_args()

# def get_pairs_generation(pairs_generation):
#     if(pairs_generation == "dp"):
#         return PapersPairBuilder.Pairs_Generation.DUPLICATED_PAIRS
#     elif (pairs_generation == "ocp"):
#         return PapersPairBuilder.Pairs_Generation.ONE_CLASS_PAIRS
#     elif (pairs_generation == "edp"):
#         return PapersPairBuilder.Pairs_Generation.EQUALLY_DISTRIBUTED_PAIRS
#
# def get_model_training(model_generation):
#     if (model_generation == "sm"):
#         return LearningToRank.Model_Training.SINGLE_MODEL_ALL_USERS
#     elif (model_generation == "mpu"):
#         return LearningToRank.Model_Training.MODEL_PER_USER

spark = SparkSession.builder.appName("LTRRecommender").config("spark.jars", "/home/polina/Desktop/LTR.jar").getOrCreate()

loader = Loader(args.input, spark)

# Only needed when folds are created
# load papers
# papers = loader.load_papers("papers_metadata.csv")
#
# # Loading of the (citeulike paper id - paper id) mapping
# # format (citeulike_paper_id, paper_id)
papers_mapping = loader.load_papers_mapping("citeulike_id_doc_id_map.csv")

# # Loading history
history = loader.load_history("ratings.csv")

# Loading bag of words for each paper
# format (paper_id, term_occurrences)
bag_of_words = loader.load_bag_of_words_per_paper("mult.dat")

# pairs_generation = get_pairs_generation(args.pairs_generation)
# model_training = get_model_training(args.model_training)
fold_validator = FoldValidator(bag_of_words, peer_papers_count=args.peers_count,
                               pairs_generation=args.pairs_generation,
                               paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id",
                               userId_col="user_id", tf_map_col="term_occurrence",
                               model_training=args.model_training)
#fold_validator.create_folds(history, papers_mapping, "new-dataset-folds-statistics.txt", timestamp_col="timestamp", fold_period_in_months=6)
fold_validator.evaluate_folds(spark)

#fold_validator.create_folds(history, papers, papers_mapping, "new-dataset-folds-statistics.txt", timestamp_col="timestamp", fold_period_in_months=1)