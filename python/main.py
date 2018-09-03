import argparse
from pyspark.sql import SparkSession

from fold_utils import FoldValidator
from loader import Loader
from logger import Logger

# make sure pyspark tells workers to use python2.7 instead of 3 if both are installed
# uncomment only during development
# os.environ['PYSPARK_PYTHON'] = '/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/System/Library/Frameworks/Python.framework/Versions/2/7/bin/python2.7'

parser = argparse.ArgumentParser(description='Process parameters needed to run the program.')
parser.add_argument('-input', type=str, required=True,
                    help='folder from where input files are read')
parser.add_argument('-peers_count', type=int, default=25,
                    help='number of peer papers generated for a paper')
parser.add_argument('-pairs_generation', type=str, default="edp",
                    help='Approaches for generating pairs. Possible options: 1) duplicated_pairs - dp , 2) one_class_pairs - ocp, 3) equally_distributed_pairs - edp')
parser.add_argument('-model_training', type=str, default="cmp",
                    help='Different training approaches for LTR. Possible options 1) general model - gm 2) individual model parallel version - imp 3) individual model squential version - ims 4) cmp - clustered model')
args = parser.parse_args()

spark = SparkSession.builder.appName("Multi-model_SVM").getOrCreate()

Logger.log(" Model training:" + str(args.model_training) + ". Peers count:" + str(args.peers_count) + ". Pairs Method:" + str(args.pairs_generation))
Logger.log("Loading the data.")

loader = Loader(args.input, spark)

# Only needed when folds are created
# load papers
# papers = loader.load_papers("papers_metadata.csv")

# Loading of the (citeulike paper id - paper id) mapping
# format (citeulike_paper_id, paper_id)
papers_mapping = loader.load_papers_mapping("citeulike_id_doc_id_map.csv")

# Loading history
history = loader.load_history("ratings.csv")

# Loading bag of words for each paper
# format (paper_id, term_occurrences)
bag_of_words = loader.load_bag_of_words_per_paper("mult.dat")
Logger.log("Loading completed.")

fold_validator = FoldValidator(peer_papers_count=args.peers_count,
                               pairs_generation=args.pairs_generation,
                               paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id",
                               userId_col="user_id", tf_map_col="term_occurrence",
                               model_training=args.model_training)

# uncomment if you need to create folds
# fold_validator.create_folds(spark, history, bag_of_words, papers_mapping, "new-dataset-folds-statistics.txt", timestamp_col="timestamp", fold_period_in_months=6)
fold_validator.evaluate_folds(spark)
