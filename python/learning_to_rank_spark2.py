from pyspark.ml.base import Estimator
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy
from pyspark.mllib.util import MLUtils
import pyspark.sql.functions as F
from random import randint
from pyspark.ml.base import Transformer
from random import shuffle
from LTR_SVM_spark2 import UserLabeledPoint
from pyspark.mllib.linalg import VectorUDT
from pyspark.mllib.linalg import Vectors
from LTR_SVM_spark2 import LTRSVMWithSGD
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, StringType, FloatType, Row


class PeerPapersSampler(Transformer):
    """
    Generate peer papers for a paper. For each paper in the data set, a list of peer papers will be selected from the paper corpus.
    The peer papers are selected randomly.
    """

    def __init__(self, papers_corpus, peer_papers_count, paperId_col="paper_id", userId_col="user_id",
                 output_col="peer_paper_id"):
        """
        :param papers_corpus: PaperCorpus object that contains data frame. It represents all papers from the corpus. 
        Possible format (paper_id, citeulike_paper_id). See PaperCorpus documentation for more information.
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
        The method generates each pair, a list of peer papers. Each user has a library which is a list of paper ids 
        that the user likes. Peer papers for each paper are generated randomly from all papers in the paper corpus 
        except the papers that the user in the pair(paper,user) likes. At the end, output data set will have an additional 
        column output_col that contains one of the generated ids. The number of row per pair will be equal to the 
        number of generated peer papers. For example, if a paper corpus contains paper ids - [1, 2, 3, 4, 5, 6],
        a (user, paper) pair is (1, 2). The user library (user, [list of liked papers]) is (1, [1, 2, 3]). If the number 
        of generated papers for a pair is 2, for the example, there is three possibilities (4, 5), (5, 6) or (4, 6).

        :param dataset: mandatory column paperId_col
        :return: data set with additional column output_col
        """

        # because paper ids in the papers_corpus are not sequential, generate a column "paper_id_index" with sequential order
        papers = self.papers_corpus.papers.drop(self.papers_corpus.citeulikePaperId_col)

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
    be computed and added wirh class 1. And for the other 50%, (p_p - p) with class 0.
    """

    # class Pairs_Generation(Enum):
    #     """
    #     Enum that contains different approaches for generating pairs.
    #     There are three possible values: duplicated_pairs, one_class_pairs,
    #     equally_distributed_pairs. For example, if we have a paper p, and a set of peer papers N for the paper p
    #     1) DUPLICATED_PAIRS - for each paper p_p of the set N, calculate (p - p_p, class:1) and
    #         (p_p - p, class: 0)
    #     2) ONE_CLASS_PAIRS - for each paper p_p of the set N, calculate (p - p_p, class:1)
    #     3) EQUALLY_DISTRIBUTED_PAIRS - for 50% of papers in the set N, calculate (p - p_p, class:1),
    #     and for the other 50% (p_p - p, class: 0)
    #     """
    #     DUPLICATED_PAIRS = 0
    #     ONE_CLASS_PAIRS = 1
    #     EQUALLY_DISTRIBUTED_PAIRS = 2

    def __init__(self, pairs_generation, model_training, userId_col="user_id", paperId_col="paper_id", peer_paperId_col="peer_paper_id",
                 paper_vector_col="paper_vector", peer_paper_vector_col="peer_paper_vector",
                 output_col="pair_paper_difference", label_col="label"):
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
        :param model_training sm, smmu or mpu
        :param userId_col name of the user id column in the input data frame
        :param paperId_col name of the column that contains paper id
        :param peer_paperId_col name of the column that contains peer paper id
        :param paper_vector_col name of the column that contains representation of the paper
        :param peer_paper_vector_col name of the column that contains representation of the peer paper
        :param output_col: name of the column where the result vector is stored
        :param label_col name of the column where the class of the pair/result is stored
        """
        self.pairs_generation = pairs_generation
        self.model_training = model_training
        self.paperId_col = paperId_col
        self.userId_col = userId_col
        self.peer_paperId_col = peer_paperId_col
        self.paper_vector_col = paper_vector_col
        self.peer_paper_vector_col = peer_paper_vector_col
        self.output_col = output_col
        self.label_col = label_col

    def _transform(self, dataset):

        # dataset format peer_paper_id | paper_id | user_id | citeulike_paper_id | lda_vector | peer_paper_lda_vector |
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
            if(self.model_training == "sm"):
                # get a list of peer paper ids per paper
                dataset = dataset.select(self.paperId_col, self.peer_paperId_col, self.paper_vector_col, self.peer_paperId_col, self.peer_paper_vector_col).dropDuplicates()
                peers_per_paper = dataset.select(self.paperId_col, self.peer_paperId_col).dropDuplicates()
                peers_per_paper = peers_per_paper.groupBy(self.paperId_col).agg(F.collect_list(self.peer_paperId_col).alias("peers_per_paper"))
            else:
                peers_per_paper = dataset.groupBy(self.userId_col, self.paperId_col).agg(F.collect_list(self.peer_paperId_col).alias("peers_per_paper"))

            # generate 50/50 distribution to positive/negative class
            peers_per_paper = peers_per_paper.withColumn("equally_distributed_papers", split_papers_udf("peers_per_paper"))
            # positive label 1
            positive_class_per_paper = peers_per_paper.withColumn("positive_class_papers", F.col("equally_distributed_papers")[0])

            if(self.model_training == "sm"):
                positive_class_per_paper = positive_class_per_paper.select(self.paperId_col,
                                                                           F.explode("positive_class_papers").alias(
                                                                               self.peer_paperId_col))
                positive_class_dataset = dataset.join(positive_class_per_paper,
                                                      [self.paperId_col, self.peer_paperId_col])
            else:
                positive_class_per_paper = positive_class_per_paper.select(self.userId_col, self.paperId_col, F.explode("positive_class_papers").alias(self.peer_paperId_col))
                positive_class_dataset = dataset.join(positive_class_per_paper, [self.userId_col, self.paperId_col, self.peer_paperId_col])

            # add the difference (paper_vector - peer_paper_vector) with label 1
            positive_class_dataset = positive_class_dataset.withColumn(self.output_col, vector_diff_udf(self.paper_vector_col, self.peer_paper_vector_col))
            # add label 1
            positive_class_dataset = positive_class_dataset.withColumn(self.label_col, F.lit(1))


            # negative label 0
            negative_class_per_paper = peers_per_paper.withColumn("negative_class_papers", F.col("equally_distributed_papers")[1])
            if (self.model_training == "sm"):
                negative_class_per_paper = negative_class_per_paper.select(self.paperId_col, F.explode("negative_class_papers").alias(
                                                                               self.peer_paperId_col))
                negative_class_dataset = dataset.join(negative_class_per_paper, [self.paperId_col, self.peer_paperId_col])
            else:
                negative_class_per_paper = negative_class_per_paper.select(self.userId_col, self.paperId_col,
                                                                           F.explode("negative_class_papers").alias(
                                                                               self.peer_paperId_col))
                negative_class_dataset = dataset.join(negative_class_per_paper,
                                                      [self.userId_col, self.paperId_col, self.peer_paperId_col])

            # add the difference (peer_paper_vector - paper_vector) with label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.output_col, vector_diff_udf( self.peer_paper_vector_col, self.paper_vector_col))
            # add label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.label_col, F.lit(0))
            dataset = positive_class_dataset.union(negative_class_dataset)
        elif (self.pairs_generation == "dp"): #self.Pairs_Generation.DUPLICATED_PAIRS):

            # vector_diff_udf = F.udf(diff, VectorUDT())
            # add the difference (paper_vector - peer_paper_vector) with label 1
            positive_class_dataset = dataset.withColumn(self.output_col, vector_diff_udf( self.paper_vector_col, self.peer_paper_vector_col))
            # add label 1
            positive_class_dataset = positive_class_dataset.withColumn(self.label_col, F.lit(1))

            # add the difference (peer_paper_vector - paper_vector) with label 0
            negative_class_dataset = dataset.withColumn(self.output_col, vector_diff_udf(self.peer_paper_vector_col, self.paper_vector_col))
            # add label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.label_col, F.lit(0))

            dataset = positive_class_dataset.union(negative_class_dataset)
        elif (self.pairs_generation == "ocp"): #self.Pairs_Generation.ONE_CLASS_PAIRS):
            # vector_diff_udf = F.udf(diff, VectorUDT())
            # add the difference (paper_vector - peer_paper_vector) with label 1
            dataset = dataset.withColumn(self.output_col, vector_diff_udf(self.paper_vector_col, self.peer_paper_vector_col))
            # add label 1
            dataset = dataset.withColumn(self.label_col, F.lit(1))
        else:
            # throw an error - unsupported option
            raise ValueError('The option' + self.pairs_generation + ' is not supported.')

        return dataset


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

        3.1) MODEL_PER_USER - Train one model per user. Number of models is equal to number of users. Each model will be trained only over papers that are 
    liked by particular user. 

        3.2) SINGLE_MODEL_ALL_USERS  - Train only one model based on all users. The model won't be so personalized, we will have more
    general model. But the cost of training is less compared to MODEL_PER_USER.
        3.3) TODO add for SMMU single model multiple users

    """

    # class Model_Training(Enum):
    #     """
    #     Enum that contains different training approaches for LTR.
    #     """
    #
    #     """ Train one model per user. Number of models is equal to number of users.
    #     Each model will be trained only over papers that are liked by particular user. """
    #     MODEL_PER_USER = 0
    #
    #     """
    #     Train only one model based on all users. The model won't be so personalized, we will have more
    #     general model. But the cost of training is less compared to MODEL_PER_USER.
    #     """
    #     SINGLE_MODEL_ALL_USERS = 1

    def __init__(self, spark, papers_corpus, paper_profiles_model, model_training= "sm",
                 pairs_generation= "edp" , peer_papers_count=10,
                 paperId_col="paper_id", userId_col="user_id", features_col="features"):
        """
        Construct Learning-to-Rank object.

        :param spark spark instance, used for creating a data frame of user ids
        :param papers_corpus: PapersCorpus object that contains data frame. It represents all papers from the corpus. 
        Possible format (paper_id, citeulike_paper_id). See PaperCorpus documentation for more information. It is used
        during the first phase of the algorihtm when sampling of peer papers is done.

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

        :param model_training: 
        1) MODEL_PER_USER - Train one model per user. Number of models is equal to number of users.
        Each model will be trained only over papers that are liked by particular user. 

        2) SINGLE_MODEL_ALL_USERS = 1 - Train only one model based on all users. The model won't be 
        so personalized, we will have more general model. But the cost of training is less compared to MODEL_PER_USER.
        3) TODO add the third option

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
        self.peer_papers_count = peer_papers_count
        self.paperId_col = paperId_col
        self.userId_col = userId_col
        self.features_col = features_col
        self.model_training = model_training
        # user_id, model per user
        self.models = {}

    def _fit(self, dataset):
        """
        Train a SVM model/models based on the input data. Dataset contains (at least) (user, paper) pairs.
        Each paper identifier is in paperId_col. Each user identifier is in userId_col. 
        See class comments from more details about the functionality of the method.

        :return: a trained learning-to-rank model(s) that can be used for predictions
        """
        # train multiple models, one for each user in the data set
        if (self.model_training == "mpu"): #self.Model_Training.MODEL_PER_USER):
            # extract all distinct users
            distinct_user_ids = dataset.select(self.userId_col).distinct().collect()
            # train a model for each user, simply by for loop over all users
            for userId in distinct_user_ids:
                # select only records for particular user, only those papers liked by the current user
                unique_user_condition = self.userId_col + "==" + str(userId[0])
                user_dataset = dataset.filter(unique_user_condition)
                # Fit the model over full data set and produce only one model for all users
                user_lsvcModel = self.train_single_SVM_model(user_dataset)
                # add the model for the user
                self.models[userId] = user_lsvcModel
            return self.models
        # Fit the model over full dataset and produce only one model for all users
        elif (self.model_training == "sm"): # self.Model_Training.SINGLE_MODEL_ALL_USERS):
            lsvcModel = self.train_single_SVM_model(dataset)
            self.models[0] = lsvcModel
            return self.models
        elif (self.model_training == "smmu"):# self.Model_Training.SINGLE_MODEL_MULTIPLE_USERS):
            # Fit the model over full data set and produce only one model,
            # it contains a map, which has an entry for each user
            lsvcModel = self.train_single_SVM_model(dataset)
            self.models[0] = lsvcModel
            return self.models
        else:
            # throw an error - unsupported option
            raise ValueError('The option' + self.model_training + ' is not supported.')

    def train_single_SVM_model(self, dataset):
        """
        Train a single model using SVM (Support Vector Machine) algorithm.

        :param dataset: paper ids used for training
        :return: a SVM model
        """

        print("Train a single model.")
        # 1) Peer papers sampling
        nps = PeerPapersSampler(self.papers_corpus, self.peer_papers_count, paperId_col=self.paperId_col,
                                userId_col=self.userId_col,
                                output_col="peer_paper_id")
        # schema -> user_id | citeulike_paper_id | paper_id | peer_paper_id |
        dataset = nps.transform(dataset)

        # add lda paper representation to each paper based on its paper_id
        dataset = self.paper_profiles_model.transform(dataset)

        # get in which columns the result of the transform is stored
        former_paper_output_column = self.paper_profiles_model.output_col
        former_papeId_column = self.paper_profiles_model.paperId_col

        # add lda ids paper representation for peer papers
        self.paper_profiles_model.setPaperIdCol("peer_paper_id")
        self.paper_profiles_model.setOutputCol("peer_paper_lda_vector")

        # schema -> peer_paper_id | paper_id | user_id | citeulike_paper_id | lda_vector | peer_paper_lda_vector
        dataset = self.paper_profiles_model.transform(dataset)

        # 2) pair building
        # peer_paper_lda_vector, paper_lda_vector
        papersPairBuilder = PapersPairBuilder(self.pairs_generation, self.model_training, self.userId_col, self.paperId_col,
                                              peer_paperId_col="peer_paper_id",
                                              paper_vector_col=former_paper_output_column,
                                              peer_paper_vector_col="peer_paper_lda_vector",
                                              output_col=self.features_col, label_col="label")
        # paper_id | peer_paper_id | user_id | citeulike_paper_id | lda_vector | peer_paper_lda_vector | features | label
        dataset = papersPairBuilder.transform(dataset)

        # drop lda vectors - not needed anymore
        dataset = dataset.drop("peer_paper_lda_vector", "lda_vector")
        lsvcModel = None
        if (self.model_training == "mpu" or self.model_training == "sm"):
            # create Label Points needed for the model
            def createLabelPoint(line):
                # label, features
                # paper_id | peer_paper_id | user_id | citeulike_paper_id | features | label
                return LabeledPoint(line[-1], line[-2])

            # convert data points data frame to RDD
            labeled_data_points = dataset.rdd.map(createLabelPoint)
            # Build the model
            lsvcModel = SVMWithSGD().train(labeled_data_points)

        elif (self.model_training == "smmu"):
            # create User Labeled Points needed for the model
            def createUserLabeledPoint(line):
                # paper_id | peer_paper_id | user_id | citeulike_paper_id | features | label
                # userId, label, features
                return UserLabeledPoint(int(line[2]), line[5], line[4])

            # convert data points data frame to RDD
            labeled_data_points = dataset.rdd.map(createUserLabeledPoint)

            # Build the model
            lsvcModel = LTRSVMWithSGD().train(labeled_data_points)

        # return the default columns of the paper profiles model, the model is ready for the training
        # of the next SVM model
        self.paper_profiles_model.setPaperIdCol(former_papeId_column)
        self.paper_profiles_model.setOutputCol(former_paper_output_column)

        print("Training LTRModel finished.")
        return lsvcModel

    def _transform(self, papers_corpus):
        """
        Add prediction to each paper in the input data set based on the trained model and its features vector.

        :param dataset: paper profiles
        :return: dataset with predictions - column "prediction"
        """
        if (self.model_training == "mpu"): #self.Model_Training.MODEL_PER_USER):
            papers_corpus_predictions = None

            self.paper_profiles_model.setPaperIdCol(self.paperId_col)
            self.paper_profiles_model.setOutputCol(self.features_col)
            # add paper representation to each paper in the corpus
            papers_corpus = self.paper_profiles_model.transform(papers_corpus)

            for userId, model in self.models.items():
                # TODO optimize by removing papers from the training set
                # make predictions using the model over full papers corpus
                papers_corpus = MLUtils.convertVectorColumnsFromML(papers_corpus, self.features_col)

                # set threshold to NONE to receive raw predictions from the model
                model._threshold = None
                user_papers_corpus_predictions_rdd = papers_corpus.rdd.map(
                    lambda p: (p.paper_id, p.citeulike_paper_id, float(model.predict(p.features))))

                # convert RDD to Data frame
                # Load papers into DF
                prediction_scheme = StructType([
                    #  name, dataType, nullable
                    StructField("paper_id", IntegerType(), False),
                    StructField("citeulike_paper_id", StringType(), True),
                    StructField("ranking_score", FloatType(), True)
                ])
                user_papers_corpus_predictions = user_papers_corpus_predictions_rdd.toDF(prediction_scheme)

                # add user id to each row to distinguish which model was used for these predictions
                user_id_df = self.spark.createDataFrame([(userId)], [self.userId_col])
                user_papers_corpus_predictions = user_papers_corpus_predictions.crossJoin(user_id_df)

                # add predictions for a user to the final set of predictions
                if (papers_corpus_predictions == None):
                    papers_corpus_predictions = user_papers_corpus_predictions
                else:
                    papers_corpus_predictions = papers_corpus_predictions.union(user_papers_corpus_predictions)

        elif (self.model_training == "sm"): #self.Model_Training.SINGLE_MODEL_ALL_USERS):
            model = self.models[0]
            self.paper_profiles_model.setPaperIdCol(self.paperId_col)
            self.paper_profiles_model.setOutputCol(self.features_col)

            # add paper representation to each paper in the corpus
            papers_corpus = self.paper_profiles_model.transform(papers_corpus)

            # make predictions using the model over full papers corpus
            # TODO check if this conversion is still needed
            papers_corpus = MLUtils.convertVectorColumnsFromML(papers_corpus, self.features_col)

            # set threshold to NONE to receive raw predictions from the model
            model._threshold = None
            papers_corpus_predictions = papers_corpus.rdd.map(lambda p: (p.paper_id, p.citeulike_paper_id, float(model.predict(p.features))))

            # convert RDD to Data frame
            # Load papers into DF
            prediction_scheme = StructType([
                #  name, dataType, nullable
                StructField("paper_id", IntegerType(), False),
                StructField("citeulike_paper_id", StringType(), True),
                StructField("ranking_score", FloatType(), True)
            ])
            papers_corpus_predictions = papers_corpus_predictions.toDF(prediction_scheme)

        elif (self.model_training == "smmu"):
            print("Predicting smmu...")
            model = self.models[0]

            self.paper_profiles_model.setPaperIdCol(self.paperId_col)
            self.paper_profiles_model.setOutputCol(self.features_col)
            # add paper representation to each paper in the corpus
            papers_corpus = self.paper_profiles_model.transform(papers_corpus)

            # TODO optimize by removing papers from the training set
            # make predictions using the model over full papers corpus
            papers_corpus = MLUtils.convertVectorColumnsFromML(papers_corpus, self.features_col)

            userIds = []
            for key in model.modelWeights:
                keySt = str(key)
                userIdRow = Row(user_id=keySt)
                userIds.append(userIdRow)

            print("Users:" * str(len(userIds)))

            # add user id to each row to distinguish which model was used for these predictions
            user_ids_df = self.spark.createDataFrame(userIds)
            user_papers_corpus = papers_corpus.crossJoin(user_ids_df)

            # set threshold to NONE to receive raw predictions from the model
            model.threshold = None
            user_papers_corpus_predictions_rdd = user_papers_corpus.rdd.map(lambda p: (
                    p.paper_id, p.citeulike_paper_id, p.user_id, float(model.predict(p.user_id, p.features))))

            print("Adding predictions")
            # convert RDD to Data frame
            prediction_scheme = StructType([
                #  name, dataType, nullable
                StructField("paper_id", IntegerType(), False),
                StructField("citeulike_paper_id", StringType(), True),
                StructField("user_id", StringType(), True),
                StructField("ranking_score", FloatType(), True)
            ])
            papers_corpus_predictions = user_papers_corpus_predictions_rdd.toDF(prediction_scheme)

        else:
            # throw an error - unsupported option
            raise ValueError('The option' + self.model_training + ' is not supported.')
        print("Prediction size:")
        print(papers_corpus_predictions.count())
        # paper_id | citeulike_paper_id| user_id | ranking_score|
        return papers_corpus_predictions