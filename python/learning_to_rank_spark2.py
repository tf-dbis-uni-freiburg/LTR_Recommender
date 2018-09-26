from pyspark.ml.base import Estimator
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy
import time
import csv
import datetime
import threading
from pyspark.mllib.util import MLUtils
import pyspark.sql.functions as F
from random import randint
from pyspark.ml.base import Transformer
from random import shuffle
from LTR_SVM_spark2 import UserLabeledPoint
from pyspark.mllib.linalg import VectorUDT
from pyspark.mllib.linalg import Vectors
from LTR_SVM_spark2 import LTRSVMWithSGD
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, StringType, FloatType, Row, DoubleType
import sys
from logger import Logger
import os

class PeerPapersSampler(Transformer):
    """
    Generate peer papers for a paper. For each paper in the data set, a list of peer papers will be selected from the paper corpus.
    The peer papers are selected randomly.
    """

    def __init__(self, papers_corpus, peer_papers_count, paperId_col="paper_id", userId_col="user_id",
                 output_col="peer_paper_id"):
        """
        :param papers_corpus: PaperCorpus object that contains data frame. It represents all papers from the corpus. 
        Possible format (paper_id) See PaperCorpus documentation for more information.
        :param peer_papers_count: number of peer papers that will be sampled per paper
        :param paperId_col name of the paper id column in the input data frame of transform()
        :param userId_col name of the user id column in the input data frame of transform()
        :param output_col the name of the column in which the produced result is stored
        """
        self.papers_corpus = papers_corpus
        self.peer_papers_count = peer_papers_count
        self.paperId_col = paperId_col
        self.userId_col = userId_col
        self.output_col = output_col

    def _transform(self, dataset):
        """
        The input data set consists of (paper, user) pairs. Each paper is represented by paper id, each user by user id.
        The names of the columns that store them are paperId_col and userId_col, respectively.
        The method generates for each pair, a list of peer papers. Each user has a library which is a list of paper ids 
        that the user likes. Peer papers for each paper are generated randomly from all papers in the paper corpus 
        except the papers that the user in the pair (paper,user) likes. At the end, output data set will have an additional
        column output_col that contains one of the generated ids. The number of row per pair will be equal to the 
        number of generated peer papers. For example, if a paper corpus contains paper ids - [1, 2, 3, 4, 5, 6],
        a (user, paper) pair is (1, 2). The user library (user, [list of liked papers]) is (1, [1, 2, 3]). If the number 
        of generated papers for a pair is 2, for the example, there is three possibilities (4, 5), (5, 6) or (4, 6).

        :param dataset: mandatory column paperId_col
        :return: data set with additional column output_col
        """

        # because paper ids in the papers_corpus are not sequential, generate a column "paper_id_index" with sequential order
        papers = self.papers_corpus.papers

        # generate sequential ids for papers, use zipWithIndex to generate ids starting from 0
        # (name, dataType, nullable)
        paper_corpus_schema = StructType([StructField("paper_id", IntegerType(), False),
                                          StructField("paper_id_index", IntegerType(), False)])
        indexed_papers_corpus = papers.rdd.zipWithIndex().map(lambda x: (int(x[0][0]), x[1])).toDF(paper_corpus_schema)

        # add the generated index to each paper in the input data set
        indexed_dataset = dataset.join(indexed_papers_corpus,
                                       indexed_papers_corpus[self.papers_corpus.paperId_col] == dataset[
                                           self.paperId_col]).drop(self.papers_corpus.paperId_col)

        total_papers_count = indexed_papers_corpus.count()

        # collect positive papers per user
        user_library = indexed_dataset.select(self.userId_col, "paper_id_index").groupBy(self.userId_col) \
            .agg(F.collect_list("paper_id_index").alias("indexed_positive_papers_per_user"))

        # add list of positive papers per user
        dataset = dataset.join(user_library, self.userId_col)

        # "indexed_positive_papers_per_user"
        # generate peer papers
        def generate_peers(positives, total_papers_count, k):
            """
            Generate peers papers for a paper. For example, if a total number of paper is 6, means that in the paper corpus these are the possible paper ids [1, 2, 3, 4, 5, 6].
            It randomly selects k of them. None of the selected id has to be in "positives" list.

            :param positives: list of paper ids. The intersection of the positives and the generated peers has to be empty.
            :param total_papers_count: total number of papers in the paper corpus
            :param k: how many peer papers have to be generated
            :return: a list of paper ids corresponding to peer papers for a paper
            """
            peers = set()
            while len(peers) < k:
                candidate = randint(1, total_papers_count + 1)
                # if a candidate paper is not in positives for an user and there exists paper with such id in the paper corpus
                # and such paper is not already part of the peers
                if candidate not in positives and candidate not in peers:
                    peers.add(candidate)
            return list(peers)

        generate_peers_udf = F.udf(generate_peers, ArrayType(IntegerType()))
        dataset = dataset.withColumn("indexed_peer_papers_per_user", generate_peers_udf("indexed_positive_papers_per_user", F.lit(total_papers_count), F.lit(self.peer_papers_count)))

        # drop columns that we have added
        dataset = dataset.drop("indexed_positive_papers_per_user")

        # explode peer paper ids column
        dataset = dataset.withColumn("indexed_peer_paper_id", F.explode("indexed_peer_papers_per_user")) \
            .drop("indexed_peer_papers_per_user")

        # rename the column before the join to not mix them
        dataset = dataset.withColumnRenamed(self.paperId_col, "positive_paper_id")

        # revert indexed_peer_paper_id to original paper_id
        dataset = dataset.join(indexed_papers_corpus,
                               dataset.indexed_peer_paper_id == indexed_papers_corpus.paper_id_index)
        dataset = dataset.drop("paper_id_index", "indexed_peer_paper_id")
        dataset = dataset.withColumnRenamed(self.paperId_col, self.output_col)
        dataset = dataset.withColumnRenamed("positive_paper_id", self.paperId_col)
        return dataset

    def store_peers(self, fold_index, dataset):
        path = "distributed-fold-" + str(fold_index) + "/" + "peers.parquet"
        dataset.write.parquet(path)

    def load_peers(self, spark, fold_index, output_directory, split_method='time-aware', peer_size=1, min_sim=0):
        # Load Candidate Set
        if split_method == 'time-aware':
            path = "distributed-fold-" + str(fold_index) + "/" + "peers.parquet"
            peers = spark.read.parquet(path)
        else:
            path = os.path.join(output_directory, "{}_folds".format(split_method),"peers", "peers_{}_minsim_{}.csv".format(peer_size, int(min_sim) if min_sim ==0 else min_sim))
            schema = StructType([StructField("user_id", IntegerType(), False), StructField("paper_id", IntegerType(), False),
                                      StructField("peer_id", IntegerType(), False), StructField("similarity", FloatType(), False),
                                      StructField("user_sim", FloatType(), False)])
            # load test data frame
            peers = spark.read.csv(path, header=True, schema=schema)
        return peers

class PapersPairBuilder(Transformer):
    """
    Class that builds pairs based on the representations of papers and their peers papers These representations are in form 
    vectors with the same size. It provides also different options on build the pairs. 
    For example, if a representation of a paper is p and a representation of its corresponding peer paper is p_p. 
    There are three options (See Pairs_Generation)for pair building:
    1) DUPLICATED_PAIRS - In the given example above, the difference between (p - p_p) will be computed and added with class 1.
    As well as (p_p - p) with class 0.
    2) ONE_CLASS_PAIRS - only the difference between (p - p_p) will be computed and added with class 1.
    3) EQUALLY_DISTRIBUTED_PAIRS - because each paper p has a set of corresponding peer papers, for 50% of them (p - p_p) will 
    be computed and added with class 1. And for the other 50%, (p_p - p) with class 0.
    """


    def __init__(self, pairs_generation, pairs_features_generation_method, model_training, vectorizer_model, userId_col="user_id",
                 paperId_col="paper_id", peer_paperId_col="peer_paper_id",
                 output_col="features", label_col="label"):
        """
        Constructs the builder.


        :param pairs_generation: there are three possible values: See Pairs_Generation. 
        For example, if we have a paper p, and a set of peer papers N for the paper p
        1) DUPLICATED_PAIRS - for each paper p_p of the set N, calculate (p - p_p, class:1) and 
        (p_p - p, class: 0)
        2) ONE_CLASS_PAIRS - for each paper p_p of the set N, calculate (p - p_p, class:1)
        3) EQUALLY_DISTRIBUTED_PAIRS - for 50% of papers in the set N, calculate p - p_p, class:1), 
        and for the other 50% (p_p - p, class: 0). If we train a general model, distribution is based on paper.
        If we train an individual model, distribution is based on pairs (user_id, paper). For the former, all peers for
        a paper independently from user information are divided into 50/50.
        For the latter, peer papers for a paper liked by a particular user are divided into 50/50.
        :param pairs_features_generation_method The method used in forming the feature vector of the pair, options are:
        %TODO: specify the features
        1) sub: p-p' (default)
        2)
        3)
        :param model_training sm, smmu or mpu
        :param userId_col name of the user id column in the input data frame
        :param paperId_col name of the column that contains paper id
        :param peer_paperId_col name of the column that contains peer paper id
        :param output_col: name of the column where the result vector is stored
        :param label_col name of the column where the class of the pair/result is stored
        """
        self.pairs_generation = pairs_generation
        self.model_training = model_training
        self.vectorizer_model = vectorizer_model
        self.paperId_col = paperId_col
        self.userId_col = userId_col
        self.peer_paperId_col = peer_paperId_col
        self.output_col = output_col
        self.label_col = label_col
        self.pairs_features_generation_method= pairs_features_generation_method

    def _generate_pair_features(self, pairs_features_generation_method, dataset):
        """
        :param pairs_features_generation_method The method used in forming the feature vector of the pair, options are:
        TODO (Anas): specify the features
        1) sub: p-p' (default)
        2)
        3)
        :param dataset: dataframe, with the following schema, where 'feature' is already the substraction of p-p'
         peer_paper_id | paper_id | user_id | similarity | user_sim | features
        :return: dataframe with the structure, where the 'features' is now updated using the pairs_features_generation_method
        peer_paper_id | paper_id | user_id | similarity | user_sim |  features
        """
        if pairs_features_generation_method == 'sub':
            return dataset
        #TODO (Anas): Define the othe methods  (udf!!!!) for pairs_features_generation_method and deal with the following line
        return dataset



    def _transform(self, dataset):
        # TODO check dataframe format
        # dataset format -> peer_paper_id | paper_id | user_id
        def diff(v1, v2):
            """
            Calculate the difference between two arrays.

            :return: array of their difference
            """
            array1 = numpy.array(v1)
            array2 = numpy.array(v2)
            result = numpy.subtract(array1, array2)
            return Vectors.dense(result)

        def split_papers(papers_id_list):
            """
            Shuffle the input list of paper ids and divide it into two lists. The ratio is 50/50.

            :param: papers_id_list initial list of paper ids that will be split
            :return: two arrays with paper ids. The first one contains the "positive paper ids" or those
            which difference will be added with label 1. The second - "the negative paper ids" -  added with
            label 0.
            """
            shuffle(papers_id_list)
            ratio = int(0.5 * len(papers_id_list))
            positive_class_set = papers_id_list[:ratio]
            negative_class_set = papers_id_list[ratio:]
            return [positive_class_set, negative_class_set]

        vector_diff_udf = F.udf(diff, VectorUDT())
        split_papers_udf = F.udf(split_papers, ArrayType(ArrayType(StringType())))

        if (self.pairs_generation == "edp" ): # self.Pairs_Generation.EQUALLY_DISTRIBUTED_PAIRS):
            # 50 % of the paper_pairs with label 1, 50% with label 0
            peers_per_paper = None
            if(self.model_training == "gm"):
                # get a list of peer paper ids per paper
                dataset = dataset.select(self.paperId_col, self.peer_paperId_col).dropDuplicates()
                peers_per_paper = dataset.groupBy(self.paperId_col).agg(F.collect_list(self.peer_paperId_col).alias("peers_per_paper"))
            else:
                peers_per_paper = dataset.groupBy(self.userId_col, self.paperId_col).agg(F.collect_list(self.peer_paperId_col).alias("peers_per_paper"))

            # generate 50/50 distribution to positive/negative class
            peers_per_paper = peers_per_paper.withColumn("equally_distributed_papers", split_papers_udf("peers_per_paper"))
            # positive label 1
            # user_id | paper_id | peers_per_paper | equally_distributed_papers | positive_class_papers |
            positive_class_per_paper = peers_per_paper.withColumn("positive_class_papers", F.col("equally_distributed_papers")[0])

            # user_id | paper_id | peer_paper_id
            if (self.model_training == "gm"):
                positive_class_per_paper = positive_class_per_paper.select(self.paperId_col, F.explode("positive_class_papers").alias(self.peer_paperId_col))
            else:
                 positive_class_per_paper = positive_class_per_paper.select(self.userId_col, self.paperId_col, F.explode("positive_class_papers").alias(self.peer_paperId_col))

            # add lda paper representation to each paper based on its paper_id
            positive_class_dataset = self.vectorizer_model.transform(positive_class_per_paper)
            # get in which columns the result of the transform is stored
            former_paper_output_column = self.vectorizer_model.output_col
            former_papeId_column = self.vectorizer_model.paperId_col

            # add lda ids paper representation for peer papers
            self.vectorizer_model.setPaperIdCol(self.peer_paperId_col)
            self.vectorizer_model.setOutputCol("peer_paper_lda_vector")

            # schema -> peer_paper_id | paper_id | user_id | lda_vector | peer_paper_lda_vector
            positive_class_dataset = self.vectorizer_model.transform(positive_class_dataset)

            # return the default columns of the paper profiles model, the model is ready for the training
            # of the next SVM model
            self.vectorizer_model.setPaperIdCol(former_papeId_column)
            self.vectorizer_model.setOutputCol(former_paper_output_column)

            # add the difference (paper_vector - peer_paper_vector) with label
            positive_class_dataset = positive_class_dataset.withColumn(self.output_col, vector_diff_udf(former_paper_output_column, "peer_paper_lda_vector"))

            # add label 1
            positive_class_dataset = positive_class_dataset.withColumn(self.label_col, F.lit(1))

            # negative label 0
            negative_class_per_paper = peers_per_paper.withColumn("negative_class_papers", F.col("equally_distributed_papers")[1])
            if (self.model_training == "gm"):
                negative_class_per_paper = negative_class_per_paper.select(self.paperId_col,
                                                                           F.explode("negative_class_papers").alias(self.peer_paperId_col))
            else:
                negative_class_per_paper = negative_class_per_paper.select(self.userId_col, self.paperId_col,
                                                                           F.explode("negative_class_papers").alias(self.peer_paperId_col))

            # add lda paper representation to each paper based on its paper_id
            negative_class_dataset = self.vectorizer_model.transform(negative_class_per_paper)
            # get in which columns the result of the transform is stored
            former_paper_output_column = self.vectorizer_model.output_col
            former_papeId_column = self.vectorizer_model.paperId_col

            # add lda ids paper representation for peer papers
            self.vectorizer_model.setPaperIdCol(self.peer_paperId_col)
            self.vectorizer_model.setOutputCol("peer_paper_lda_vector")

            # schema -> peer_paper_id | paper_id | user_id | lda_vector | peer_paper_lda_vector
            negative_class_dataset = self.vectorizer_model.transform(negative_class_dataset)

            # return the default columns of the paper profiles model, the model is ready for the training
            # of the next SVM model
            self.vectorizer_model.setPaperIdCol(former_papeId_column)
            self.vectorizer_model.setOutputCol(former_paper_output_column)

            # add the difference (peer_paper_vector - paper_vector) with label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.output_col, vector_diff_udf("peer_paper_lda_vector", former_paper_output_column))
            # add label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.label_col, F.lit(0))

            result = positive_class_dataset.union(negative_class_dataset)

        elif (self.pairs_generation == "dp"): #self.Pairs_Generation.DUPLICATED_PAIRS):

            # add lda paper representation to each paper based on its paper_id
            dataset = self.vectorizer_model.transform(dataset)
            # get in which columns the result of the transform is stored
            former_paper_output_column = self.vectorizer_model.output_col
            former_papeId_column = self.vectorizer_model.paperId_col

            # add lda ids paper representation for peer papers
            self.vectorizer_model.setPaperIdCol(self.peer_paperId_col)
            self.vectorizer_model.setOutputCol("peer_paper_lda_vector")

            # schema -> peer_paper_id | paper_id | user_id ? | lda_vector | peer_paper_lda_vector
            dataset = self.vectorizer_model.transform(dataset)

            # return the default columns of the paper profiles model, the model is ready for the training
            # of the next SVM model
            self.vectorizer_model.setPaperIdCol(former_papeId_column)
            self.vectorizer_model.setOutputCol(former_paper_output_column)

            # add the difference (paper_vector - peer_paper_vector) with label 1
            positive_class_dataset = dataset.withColumn(self.output_col, vector_diff_udf(former_paper_output_column, "peer_paper_lda_vector"))
            # add label 1
            positive_class_dataset = positive_class_dataset.withColumn(self.label_col, F.lit(1))
            # add the difference (peer_paper_vector - paper_vector) with label 0
            negative_class_dataset = dataset.withColumn(self.output_col, vector_diff_udf("peer_paper_lda_vector", former_paper_output_column))
            # add label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.label_col, F.lit(0))
            result = positive_class_dataset.union(negative_class_dataset)
        elif (self.pairs_generation == "ocp"): #self.Pairs_Generation.ONE_CLASS_PAIRS):

            # add lda paper representation to each paper based on its paper_id
            dataset = self.vectorizer_model.transform(dataset)
            # get in which columns the result of the transform is stored
            former_paper_output_column = self.vectorizer_model.output_col
            former_papeId_column = self.vectorizer_model.paperId_col

            # add lda ids paper representation for peer papers
            self.vectorizer_model.setPaperIdCol(self.peer_paperId_col)
            self.vectorizer_model.setOutputCol("peer_paper_lda_vector")

            # schema -> peer_paper_id | paper_id | user_id ? | lda_vector | peer_paper_lda_vector
            dataset = self.vectorizer_model.transform(dataset)

            # return the default columns of the paper profiles model, the model is ready for the training
            # of the next SVM model
            self.vectorizer_model.setPaperIdCol(former_papeId_column)
            self.vectorizer_model.setOutputCol(former_paper_output_column)

            # add the difference (paper_vector - peer_paper_vector) with label 1
            result = dataset.withColumn(self.output_col, vector_diff_udf(former_paper_output_column, "peer_paper_lda_vector"))
            # add label 1
            result = result.withColumn(self.label_col, F.lit(1))
        else:
            # throw an error - unsupported option
            raise ValueError('The option' + self.pairs_generation + ' is not supported.')

        # drop lda vectors - not needed anymore
        result = result.drop("peer_paper_lda_vector", former_paper_output_column)

        # Manipulate the feature vector, depending on the pairs_features_generation_method, the default is sub
        result = self._generate_pair_features(self.pairs_features_generation_method, result)

        return result


class LearningToRank(Estimator, Transformer):
    """
    Class that implements different approaches of learning to rank algorithms. 
    Learning to Rank algorithm includes 3 phases:
    1) peers extraction - for each paper p in the data set, k peer papers are extracted. In the current implementation,
    peer papers are extracted randomly from {paper_corpus}\{papers liked by the same user who liked p}. If paper p is liked by
    many users, for each pair (user,p) will be extracted peer_papers_count peer papers to p and the user's library will be removed from the 
    possible set of papers used.
    2) Before phase 2, a model that can produce a representation for each paper is used. For example, TFIDFVectorizorModel can add a tf-idf vector
    based on input paper id. Such a model is used for adding a representation for each paper and its peer papers. After each pair (paper_vector, peer_paper_vector)
    is ready, PapersPairBuilder is used to: 1) subtract the two vectors and add the corresponding class. PapersPairBuilder has three options for pair building
    (See Pairs_Generation). For example, if a representation of a paper is p and a representation of its corresponding peer paper is p_p, 
    pair generation could be one of the following:
        1) DUPLICATED_PAIRS - In the given example above, the difference between (p - p_p) will be computed and added with class 1.
        As well as (p_p - p) with class 0.
        2) ONE_CLASS_PAIRS - only the difference between (p - p_p) will be computed and added with class 1.
        3) EQUALLY_DISTRIBUTED_PAIRS - because each paper p has a set of corresponding peer papers, for 50% of them (p - p_p) will 
        be computed and added wirh class 1. And for the other 50%, (p_p - p) with class 0.

    3) The last step is training a model(s) based on the produced difference of vectors and their class. At the moment, only an implementation
    of SVM (Support Vector Machines) is supported. When the model(s) is/are trained it can be used for predictions.
    There are two possibilities for model training:

        3.1) GENERAL_MODEL (gm) - Train one model per user. Number of models is equal to number of users. Each model will be trained only over papers that are
    liked by particular user. 

        3.2) INDIVIDUAL MODEL SEQUENTIAL (ims) - Train only one model based on all users. The model won't be so personalized, we will have more
    general model. But the cost of training is less compared to GENERAL_MODEL.
        3.3) INDIVIDUAL MODEL PARALLEL (imp)

    """

    def __init__(self, spark, papers_corpus, paper_profiles_model, user_clusters=None, model_training= "gm",
                 pairs_generation= "edp" , pairs_features_generation_method = "sub",
                 paperId_col="paper_id", userId_col="user_id", features_col="features"):
        """
        Construct Learning-to-Rank object.

        :param spark spark instance, used for creating a data frame of user ids
        :param papers_corpus: PapersCorpus object that contains data frame. It represents all papers from the corpus. 
        Possible format (paper_id). See PaperCorpus documentation for more information. It is used
        during the first phase of the algorithm when sampling of peer papers is done.

        :param paper_profiles_model: a model that can produce a representation for each paper is used. For example, 
        TFIDFVectorizorModel can add a tf-idf vector based on input paper id. Such a model is used for adding a
        representation for each paper and its peer papers. The transform() method implementation of the model is used.
        As well as properties like paperId_col and output_col.

        :param pairs_generation: Used during the second phase of the algorithms. There are three possible values: 
        DUPLICATED_PAIRS, ONE_CLASS_PAIRS, EQUALLY_DISTRIBUTED_PAIRS (See Pairs_generation). 
        For example, if we have a paper p, and a set of peer papers N for the paper p
        1) DUPLICATED_PAIRS - for each paper p_p of the set N, calculate (p - p_p, class:1) and 
        (p_p - p, class: 0)
        2) ONE_CLASS_PAIRS - for each paper p_p of the set N, calculate (p - p_p, class:1)
        3) EQUALLY_DISTRIBUTED_PAIRS - for 50% of papers in the set N, calculate p - p_p, class:1), 
        and for the other 50% (p_p - p, class: 0)
        :param pairs_features_generation_method TODO(Anas) add the documentation
        :param model_training: 
        1) GENERAL_MODEL - Train one model per user. Number of models is equal to number of users.
        Each model will be trained only over papers that are liked by particular user. 

        2) INDIVIDUAL MODEL SEQUENTIAL = 1 - Train only one model based on all users. The model won't be
        so personalized, we will have more general model. But the cost of training is less compared to MODEL_PER_USER.
        3) INDIVIDUAL MODEL PARALLEL

        :param peer_papers_count: Used in the first phase of the algorithm. It represents the number of peer papers that will 
        be sampled per paper

        :param paperId_col: name of a column that contains identifier of each paper in the input dataset
        :param userId_col: name of a column that contains identifier of each user in the input dataset
        :param features_col: the name of the column that contains the feature representation the model can predict on
        """
        self.spark = spark
        self.papers_corpus = papers_corpus
        self.paper_profiles_model = paper_profiles_model
        self.pairs_generation = pairs_generation
        self.paperId_col = paperId_col
        self.userId_col = userId_col
        self.features_col = features_col
        self.model_training = model_training
        # data frame with format cluster_id, centroid, [user_ids] which belong to this cluster
        self.user_clusters = user_clusters
        # user_id, model per user
        self.models = {}
        self.pairs_features_generation_method = pairs_features_generation_method

    def _fit(self, dataset):
        Logger.log("LTR:fit method")
        """
        Train a SVM model/models based on the input data. Dataset contains (at least) (user, paper) pairs.
        Each paper identifier is in paperId_col. Each user identifier is in userId_col. 
        See class comments from more details about the functionality of the method.
        
        :param dataset training data of a fold. contains user_id, paper_id. All papers
        that a user likes in the training set.
        :return: a trained learning-to-rank model(s) that can be used for predictions
        """

        papersPairBuilder = PapersPairBuilder(self.pairs_generation, self.pairs_features_generation_method, self.model_training,
                                              self.paper_profiles_model,
                                              self.userId_col, self.paperId_col,
                                              #peer_paperId_col="peer_paper_id", #TODO: for time-aware uncomment this and comment the next
                                              peer_paperId_col="peer_id",
                                              output_col=self.features_col,
                                              label_col="label")
        # 2) pair building
        # format -> paper_id | peer_paper_id | user_id | lda_vector | peer_paper_lda_vector | features | label
        dataset = papersPairBuilder.transform(dataset)
        dataset = dataset.drop("peer_paper_lda_vector", "lda_vector")
        # train multiple models, one for each user in the data set
        if (self.model_training == "ims"):

            start_time = datetime.datetime.now()
            Logger.log("Selecting users.")
            # extract all distinct users
            distinct_user_ids = dataset.select(self.userId_col).distinct().collect()
            end_time = datetime.datetime.now() - start_time
            Logger.log("Total number of users: " + str(len(distinct_user_ids)))
            Logger.log("Time for collecting users: " + str(end_time.total_seconds()))
            for userId in distinct_user_ids:
                Logger.log("Train a model for user id: " + str(userId[0]))
                start_time = datetime.datetime.now()
                # select only records for particular user, only those papers liked by the current user
                unique_user_condition = self.userId_col + "==" + str(userId[0])
                user_dataset = dataset.filter(unique_user_condition)
                lsvcModel = self.train_single_SVM_model(user_dataset)
                # add the model for the user
                Logger.log("Add key " + str(userId[0]) + " to the map.")
                self.models[int(userId[0])] = lsvcModel.weights
                time_to_train = (datetime.datetime.now() - start_time).total_seconds()
                Logger.log("Time for training for user " + str(userId[0]) + ": " + str(time_to_train))
            Logger.log("Return all trained models for users. Count:" + str(len(self.models)))
        # if we have to train multiple models based on user clustering
        elif (self.model_training == "cms"):
            Logger.log("Selecting clusters.")
            # extract all distinct clusters
            distinct_cluster_ids = self.user_clusters.select("cluster_id").distinct().collect()
            Logger.log("Total number of users: " + str(len(distinct_cluster_ids)))
            for clusterId in distinct_cluster_ids:
                Logger.log("Train a model for cluster id: " + str(clusterId[0]))
                # select only users for particular cluster
                unique_cluster_condition = "cluster_id" + "==" + str(clusterId[0])
                # cluster
                users_in_cluster = self.user_clusters.filter(unique_cluster_condition).select(self.userId_col)
                # select only those papers in the training set that are liked by users in the cluster
                cluster_dataset = dataset.join(users_in_cluster, self.userId_col)
                # Fit the model over data for a cluster based on users
                cluster_lsvcModel = self.train_single_SVM_model(cluster_dataset)
                # add the model for a cluster
                self.models[clusterId[0]] = cluster_lsvcModel
            Logger.log("Return all trained models for clusters. Count:" + str(len(self.models)))
        # Fit the model over full dataset and produce only one model for all users
        elif (self.model_training == "gm"):
            Logger.log("Train a General Model.")
            lsvcModel = self.train_single_SVM_model(dataset)
            self.models[0] = lsvcModel
        elif (self.model_training == "imp"):
            Logger.log("Train multiple models in parallel.")
            start_time = datetime.datetime.now()
            # Fit the model over full data set and produce only one model,
            # it contains a map, which has an entry for each user
            lsvcModel = self.train_single_SVM_model(dataset)
            end_time = datetime.datetime.now() - start_time
            self.models[0] = lsvcModel
            Logger.log("Time to train a IMP model over all users:" + str(end_time.total_seconds()))
            Logger.log("Return all trained models. Count:" + str(len(self.models[0].modelWeights)))
        elif (self.model_training == "cmp"):
            Logger.log("Train multiple clustered models in parallel.")
            # Fit the model over full data set and produce only one model,
            # it contains a map, which has an entry for each cluster
            lsvcModel = self.train_single_SVM_model(dataset)
            self.models[0] = lsvcModel
            Logger.log("Return all trained models. Count:" + str(len(lsvcModel.modelWeights)))
        else:
            # throw an error - unsupported option
            raise ValueError('The option' + self.model_training + ' is not supported.')
        return self.models

    def train_single_SVM_model(self, dataset):
        """
        Train a single model using SVM (Support Vector Machine) algorithm.

        :param dataset: paper ids used for training
        :return: a SVM model
        """

        Logger.log("train_single_SVM_model")
        if (self.model_training == "imp"):
            # create User Labeled Points needed for the model
            def createUserLabeledPoint(userId, label, features):
                # peer_paper_id | paper_id | user_id | features | label
                # userId, label, features
                return UserLabeledPoint(int(userId), label, features)

            createUserLabeledPoint_udf = F.udf(createUserLabeledPoint, IntegerType(), IntegerType(), ArrayType(DoubleType()))
            # convert data points data frame to RDD
            #labeled_data_points = dataset.rdd.map(createUserLabeledPoint)
            labeled_data_points = dataset.withColumn('labeledPoint',createUserLabeledPoint_udf(self.userId_col, 'label', self.features_col))\
                .select("labeledPoint").rdd
            Logger.log("Number of partitions for labeled data points: " + str(labeled_data_points.getNumPartitions()))
            # Build the model
            lsvcModel = LTRSVMWithSGD().train(labeled_data_points, intercept=False, validateData=False)
            return lsvcModel
        if (self.model_training == "cmp"):
            # select only those papers in the training set that are liked by users in the cluster
            cluster_dataset = dataset.join(self.user_clusters, self.userId_col)

            # create User Labeled Points needed for the model
            def createUserLabeledPoint(line):
                # user_id | peer_paper_id | paper_id | features | label | cluster_id |
                # clusterId, label, features
                return UserLabeledPoint(int(line[-1]), line[4], line[3])

            # convert data points data frame to RDD
            labeled_data_points = cluster_dataset.rdd.map(createUserLabeledPoint)

            # Build the model
            lsvcModel = LTRSVMWithSGD().train(labeled_data_points, validateData=False, intercept=False)
            return lsvcModel
        else:
            # create Label Points needed for the model
            def createLabelPoint(line):
                # label, features
                # paper_id | peer_paper_id | user_id | features | label
                return LabeledPoint(line[-1], line[-2])

            # convert data points data frame to RDD
            labeled_data_points = dataset.rdd.map(createLabelPoint)
            # Build the model
            lsvcModel = SVMWithSGD().train(labeled_data_points, validateData=False, intercept=False)

            return lsvcModel
        Logger.log("Training LTRModel finished.")

    def _transform(self, candidate_set):
        """
        TODO change comments
        Add prediction to each paper in the input data set based on the trained model and its features vector.

        :param dataset: paper profiles
        :return: dataset with predictions - column "prediction"
        """
        Logger.log("LTR: transform.")
        predictions = None
        # format user_id, paper_id
        candidate_set = candidate_set.select(self.userId_col, F.explode("candidate_set").alias(self.paperId_col))

        # scheme for the final prediction result data frame
        predictions_scheme = StructType([
            #  name, dataType, nullable
            StructField("user_id", StringType(), False),
            StructField("paper_id", IntegerType(), False),
            StructField("ranking_score", FloatType(), True)
        ])

        self.paper_profiles_model.setPaperIdCol(self.paperId_col)
        self.paper_profiles_model.setOutputCol(self.features_col)

        # add paper representation to each paper in the candidate set
        # candidate set format - user_id, paper_id
        predictions = self.paper_profiles_model.transform(candidate_set)
        # make predictions using the model over
        predictions = MLUtils.convertVectorColumnsFromML(predictions, self.features_col)

        if (self.model_training == "gm"): #self.Model_Training.SINGLE_MODEL_ALL_USERS):
            Logger.log("Prediction gm ...")

            model = self.models[0]
            # set threshold to NONE to receive raw predictions from the model
            model._threshold = None
            predictions = predictions.rdd.map(lambda p: (p.user_id, p.paper_id, float(model.predict(p.features))))
            predictions = predictions.toDF(predictions_scheme)

        elif (self.model_training == "imp"):
            Logger.log("Predicting imp...")
            model = self.models[0]
            # set threshold to NONE to receive raw predictions from the model
            model.threshold = None

            # broadcast weight vectors for all models
            model_br = self.spark.sparkContext.broadcast(model)

            predictions_rdd = predictions.rdd.map(lambda p: (p.user_id, p.paper_id, float(model_br.value.predict(p.user_id, p.features))))
            predictions = predictions_rdd.toDF(predictions_scheme)

        elif (self.model_training == "ims"):

            Logger.log("Predicting ims ...")

            # broadcast weight vectors for all models
            weights_br = self.spark.sparkContext.broadcast(self.models)

            def predict(id, features):
                weights = weights_br.value
                weight = weights[int(id)]
                prediction = weight.dot(features)
                return float(prediction)

            predict_udf = F.udf(predict, FloatType())

            predictions = predictions.withColumn("ranking_score", predict_udf("user_id", "features")) \
                    .select(self.userId_col, self.paperId_col, "ranking_score")

        elif (self.model_training == "cms"):
            Logger.log("Predicting cms ...")
            # add cluster id to each user - based on it, prediction are done
            users_in_cluster = self.user_clusters.withColumn(self.userId_col, F.explode("user_ids")).drop("user_ids")
            predictions = predictions.join(users_in_cluster, self.userId_col)

            for clusterId, model in self.models.items():
                # set threshold to NONE to receive raw predictions from the model
                model._threshold = None

            # broadcast weight vectors for all models
            models_br = self.spark.sparkContext.broadcast(self.models)

            def predict(id, features):
                models = models_br.value
                model = models[id]
                prediction = model.predict(features)
                return float(prediction)

            predict_udf = F.udf(predict, FloatType())

            predictions = predictions.withColumn("ranking_score", predict_udf("cluster_id", "features"))\
                .select(self.userId_col, self.paperId_col, "ranking_score")
        elif (self.model_training == "cmp"):
            # format user_id, paper_id, feature
            # add cluster id to each user - based on it, prediction are done
            predictions = predictions.join(self.user_clusters, self.userId_col)
            model = self.models[0]

            # set threshold to NONE to receive raw predictions from the model
            model.threshold = None
            # broadcast weight vectors for all models
            model_br = self.spark.sparkContext.broadcast(model)

            predictions_rdd = predictions.rdd.map(
                lambda p: (p.user_id, p.paper_id, float(model_br.value.predict(p.cluster_id, p.features))))
            predictions = predictions_rdd.toDF(predictions_scheme)
        else:
            # throw an error - unsupported option
            raise ValueError('The option' + self.model_training + ' is not supported.')
        # user_id | paper_id | ranking_score|
        return predictions