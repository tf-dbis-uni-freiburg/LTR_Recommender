import argparse
from pyspark.sql import SparkSession
import os
from fold_utils import FoldValidator
from loader import Loader
from logger import Logger

parser = argparse.ArgumentParser(description='Process parameters needed to run the program.')
parser.add_argument('--input','-i', type=str, required=True, help='folder from where input files are read')
parser.add_argument('--output_dir', '-d', type=str, help='folder to store the results, the folds and the results')
parser.add_argument("--split", "-s", choices=['user-based', 'item-based', 'time-aware'],
                    help="The split strategy: user-based, splits thse ratings of each user in train/test; time-aware: is a time aware split")
parser.add_argument('--peers_count_list', '-pc', nargs='+', type=int, default=[25], help='number of peer papers generated for a paper')
parser.add_argument('--pairs_generation','-pg', type=str, default="edp", help='Approaches for generating pairs. Possible options: 1) duplicated_pairs - dp , 2) one_class_pairs - ocp, 3) equally_distributed_pairs - edp')
parser.add_argument('--model_training','-m', type=str, default="cmp",help='Different training approaches for LTR. Possible options 1) general model - gm 2) individual model parallel version - imp 3) individual model squential version - ims 4) cmp - clustered model')
parser.add_argument('--min_peer_similarity_list', '-ms', nargs='+', type=float, help='Specify one or more minimum similarity of a paper to be considered as a peer. If this is not provided, random peers will be loaded')
parser.add_argument('--folds_list', '-f', nargs='+', type=int, default=[1, 2, 3, 4, 5], help='Specify one or more fold ids (1-based) to be tested. By default, it tests five folds. Example, to test the folds 1, 3 and 4: -f 1 3 4')
parser.add_argument('--pairs_features_generation_method','-g', choices=['sub', 'peer_sim', 'user_sim'], default='sub', help='The method used in forming the feature vector of the pair, options are:[sub,...]')

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
for peers_count in args.peers_count_list: #[1,2,10,20,50]:
    if args.min_peer_similarity_list:
        for min_sim in args.min_peer_similarity_list: #[0, 0.001, 0.01, 0.1]:
            Logger.log("Model training: {} - Min sim: {} - Peers count: {} - Pairs Method: {} - Pairs features: {}".format(args.model_training, min_sim, peers_count, args.pairs_generation, args.pairs_features_generation_method))
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
            fold_validator.evaluate_folds(spark, folds = args.folds_list)
    else:
        Logger.log("Random Peers:")
        Logger.log("Model training: {} - Peers count: {} - Pairs Method: {} - Pairs features: {}".format(args.model_training, peers_count, args.pairs_generation, args.pairs_features_generation_method))
        fold_validator = FoldValidator(peer_papers_count=peers_count,
                                       pairs_generation=args.pairs_generation,
                                       pairs_features_generation_method=args.pairs_features_generation_method,
                                       model_training=args.model_training,
                                       output_dir=args.output_dir,
                                       split_method=args.split,
                                       paperId_col="paper_id",
                                       userId_col="user_id")
        """
        # uncomment to generate new folds
        # fold_validator.create_folds(spark, history, bag_of_words, tf_map_col = "term_occurrence", timestamp_col="timestamp", fold_period_in_months=6)
        """
        # # uncomment to run evaluation
        fold_validator.evaluate_folds(spark, folds=args.folds_list)

