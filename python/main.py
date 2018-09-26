import argparse
from pyspark.sql import SparkSession
import os
from fold_utils import FoldValidator
from loader import Loader
from logger import Logger

parser = argparse.ArgumentParser(description='Process parameters needed to run the program.')
parser.add_argument('--input','-i', type=str, required=True, help='folder from where input files are read')
parser.add_argument('--output_dir', '-d', type=str, help='folder to store the results, the folds and the results')
parser.add_argument("--split", "-s", choices=['user-based', 'time-aware'],
                    help="The split strategy: user-based, splits thse ratings of each user in train/test; time-aware: is a time aware split")
parser.add_argument('--peers_count','-pc', type=int, default=25, help='number of peer papers generated for a paper')
parser.add_argument('--pairs_generation','-pg', type=str, default="edp", help='Approaches for generating pairs. Possible options: 1) duplicated_pairs - dp , 2) one_class_pairs - ocp, 3) equally_distributed_pairs - edp')
parser.add_argument('--model_training','-m', type=str, default="cmp",help='Different training approaches for LTR. Possible options 1) general model - gm 2) individual model parallel version - imp 3) individual model squential version - ims 4) cmp - clustered model')
parser.add_argument('--min_peer_similarity','-ms', type=float, default=0, help='The minimum similarity of a paper to be considered as a peer')

#TODO: (Anas): add the options of the following param
parser.add_argument('--pairs_features_generation_method','-g', choices=['sub'], default='sub', help='The method used in forming the feature vector of the pair, options are:[sub,...]')
args = parser.parse_args()

# create a folder for results if it does not exist already
result_folder_name = os.path.join(args.output_dir)
if not os.path.exists(result_folder_name):
    os.makedirs(result_folder_name)

spark = SparkSession.builder.appName("Multi-model_SVM").getOrCreate()



"""
Logger.log("Loading the data.")
loader = Loader(args.input, spark)
# Loading history
# format -> timestamp | user_id | paper_id
#history = loader.load_history("ratings.csv", "citeulike_id_doc_id_map.csv")
#history = loader.load_history("ratings.csv")

# Loading bag of words for each paper
# format -> terms_count | term_occurrence | paper_id
bag_of_words =None
#bag_of_words = loader.load_bag_of_words_per_paper("mult.dat")

Logger.log("Loading completed.")

"""
#Logger.log(" Model training:" + str(args.model_training) + ". Min sim:" + str(args.min_peer_similarity) + ". Peers count:" + str(args.peers_count) + ". Pairs Method:" + str(args.pairs_generation))
for peers_count in [1,2,10,20,50]:
    for min_sim in [0, 0.001, 0.01, 0.1]:
        if peers_count == 1 and min_sim == 0:
            continue
        Logger.log("Model training: {} - Min sim: {} - Peers count: {} - Pairs Method: {}".format(args.model_training, min_sim, peers_count, args.pairs_generation))
        fold_validator = FoldValidator(peer_papers_count = peers_count,
                                       pairs_generation = args.pairs_generation,
                                       pairs_features_generation_method = args.pairs_features_generation_method,
                                       model_training = args.model_training,
                                       output_dir = args.output_dir,
                                       split_method = args.split,
                                       paperId_col = "paper_id",
                                       userId_col = "user_id", min_peer_similarity = min_sim)
        """
        # uncomment to generate new folds
        # fold_validator.create_folds(spark, history, bag_of_words, tf_map_col = "term_occurrence", timestamp_col="timestamp", fold_period_in_months=6)
        """
        # # uncomment to run evaluation
        fold_validator.evaluate_folds(spark)
