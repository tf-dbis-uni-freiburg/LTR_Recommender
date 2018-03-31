import os
from loader import  Loader
from fold_utils import  FoldValidator
from learning_to_rank_spark2 import PapersPairBuilder, LearningToRank
from pyspark.sql import SparkSession
import argparse

# make sure pyspark tells workers to use python3 not 2 if both are installed
# uncomment only during development
#os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5'

# parse input parameters
parser = argparse.ArgumentParser(description='Process parameters needed to run the program.')
parser.add_argument('-input', type=str, required=True,
                    help='folder from where input files are read')
parser.add_argument('-peers_count', type=int, default=10,
                    help='number of peer papers generated for a paper')
parser.add_argument('-pairs_generation', type=str, default="edp",
                    help='Approaches for generating pairs. Possible options: 1) duplicated_pairs - dp , 2) one_class_pairs - ocp, 3) equally_distributed_pairs - edp')
parser.add_argument('-model_training', type=str, default="mpu",
                    help='Different training approaches for LTR. Possible options 1) model per user - mpu 2) sinle model for all users - sm')
args = parser.parse_args()

def get_pairs_generation(pairs_generation):
    if(pairs_generation == "dp"):
        return PapersPairBuilder.Pairs_Generation.DUPLICATED_PAIRS
    elif (pairs_generation == "ocp"):
        return PapersPairBuilder.Pairs_Generation.ONE_CLASS_PAIRS
    elif (pairs_generation == "edp"):
        return PapersPairBuilder.Pairs_Generation.EQUALLY_DISTRIBUTED_PAIRS

def get_model_training(model_generation):
    if (model_generation == "sm"):
        return LearningToRank.Model_Training.SINGLE_MODEL_ALL_USERS
    elif (model_generation == "mpu"):
        return LearningToRank.Model_Training.MODEL_PER_USER

spark = SparkSession.builder.appName("LTRRecommender").getOrCreate()

loader = Loader(args.input, spark);

# Only needed when folds are created
# # load papers
# papers = loader.load_papers("papers.csv")
#
# # Loading of the (citeulike paper id - paper id) mapping
# # format (citeulike_paper_id, paper_id)
# papers_mapping = loader.load_papers_mapping("citeulikeId_docId_map.dat")
#
# papers = papers.join(papers_mapping, "citeulike_paper_id")
#
# # Loading of the (citeulike user hash - user id) mapping
# # format (citeulike_user_hash, user_id)
# user_mappings = loader.load_users_mapping("citeulikeUserHash_userId_map.dat")
#
# # Loading history
# history = loader.load_history("current")
#
# # map citeulike_user_hash to user_id
# # format of history (citeulike_paper_id, citeulike_user_hash, timestamp, tag, paper_id, user_id)
# history = history.join(user_mappings, "citeulike_user_hash", "inner")

# # map citeulike_paper_id to paper_id
# history = history.join(papers_mapping, "citeulike_paper_id", "inner")

# Loading bag of words for each paper
# format (paper_id, term_occurrences)

bag_of_words = loader.load_bag_of_words_per_paper("mult.dat")

pairs_generation = get_pairs_generation(args.pairs_generation)
model_traning = get_model_training(args.model_training)
fold_validator = FoldValidator(bag_of_words, peer_papers_count=args.peers_count,
                               pairs_generation=pairs_generation,
                               paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id",
                               userId_col="user_id", tf_map_col="term_occurrence",
                               model_training=model_traning)
# fold_validator.create_folds(history, papers, papers_mapping, "folds-statistics.txt", timestamp_col="timestamp", fold_period_in_months=6)
fold_validator.evaluate_folds(spark)



