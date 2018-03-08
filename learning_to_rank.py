from pyspark.ml.base import Estimator
from pyspark.ml.classification import LinearSVC
import pyspark.sql.functions as F
from spark_utils import *
from pyspark.sql.window import Window
from pyspark.ml.base import Transformer


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
    is ready, PapersPairBuilder is used to: 1) subtract the two vectors and add the corresponding class. PapersPairBuilder has three options for pair building. 
    For example, if a representation of a paper is p and a representation of its corresponding peer paper is p_p, pair generation could be one of the following:
        1) "duplicated_pairs" - In the given example above, the difference between (p - p_p) will be computed and added with class 1.
        As well as (p_p - p) with class 0.
        2) "one_class_pairs" - only the difference between (p - p_p) will be computed and added with class 1.
        3) "equally_distributed_pairs" - because each paper p has a set of corresponding peer papers, for 50% of them (p - p_p) will 
        be computed and added wirh class 1. And for the other 50%, (p_p - p) with class 0.
    
    3) The last step is training a model based on the produced difference of vectors and their class. At the moment, only an implementation
    of SVM (Support Vector Machines) is supported. When the model is trained it can be used for predictions.
    """

    def __init__(self, papers_corpus, paper_profiles_model, pairs_generation="equally_distributed_pairs", peer_papers_count=10, paperId_col="paper_id",
                 userId_col="user_id", features_col="features"):
        """
        Construct Learning-to-Rank object.
        
        :param papers_corpus: PapersCorpus object that contains data frame. It represents all papers from the corpus. 
        Possible format (paper_id, citeulike_paper_id). See PaperCorpus documentation for more information. It is used
        during the first phase of the algorihtm when sampling of peer papers is done.
        
        :param paper_profiles_model: a model that can produce a representation for each paper is used. For example, 
        TFIDFVectorizorModel can add a tf-idf vector based on input paper id. Such a model is used for adding a
        representation for each paper and its peer papers. The transform() method implementation of the model is used.
        As well as properties like paperId_col and output_col.
         
        :param pairs_generation: Used during the second phase of the algorithms. There are three possible values: 
        duplicated_pairs, one_class_pairs, equally_distributed_pairs. For example, if we have a paper p, and a set 
        of peer papers N for the paper p
        1) if it is "duplicated_pairs" - for each paper p_p of the set N, calculate (p - p_p, class:1) and 
        (p_p - p, class: 0)
        2) if it is "one_class_pairs" - for each paper p_p of the set N, calculate (p - p_p, class:1)
        3) if it is "equally_distributed_pairs" - for 50% of papers in the set N, calculate p - p_p, class:1), 
        and for the other 50% (p_p - p, class: 0)
        
        :param peer_papers_count: Used in the first phase of the algorithm. It represents the number of peer papers that will 
        be sampled per paper
        
        :param paperId_col: name of a column that contains identifier of each paper in the input dataset
        :param userId_col: name of a column that contains identifier of each user in the input dataset
        :param features_col: the name of the column that contains the feature representation the model can predict on
        """
        self.papers_corpus = papers_corpus
        self.paper_profiles_model = paper_profiles_model
        self.pairs_generation = pairs_generation
        self.peer_papers_count = peer_papers_count
        self.paperId_col = paperId_col
        self.userId_col = userId_col
        self.features_col = features_col
        self.model = None

    def _fit(self, dataset):
        """
        The input dataset has to contains at least (paper, user) pairs. Each paper identifier is in paperId_col.
        Each user identifier is in userId_col. See class comments from more details about the functionality of the method.
        
        :return: a trained learning-to-rank model that can be used for predictions
        """
        # 1) Peer papers sampling
        nps = PeerPapersSampler(self.papers_corpus, self.peer_papers_count, paperId_col=self.paperId_col, userId_col=self.userId_col,
                                    output_col="peer_paper_id")
        dataset = nps.transform(dataset)


        # add tf paper representation to each paper based on its paper_id
        dataset = self.paper_profiles_model.transform(dataset)
        # get in which column the result of the transform is stored
        paper_output_column = self.paper_profiles_model.output_col;

        # add tf paper representation  for peer papers
        self.paper_profiles_model.setPaperIdCol("peer_paper_id")
        self.paper_profiles_model.setOutputCol("peer_paper_tf_idf_vector")
        dataset = self.paper_profiles_model.transform(dataset)

        # 2) pair building
        # peer_paper_tf_idf_vector, paper_tf_idf_vector
        papersPairBuilder = PapersPairBuilder(self.pairs_generation, paperId_col=self.paperId_col,
                                                  peer_paperId_col="peer_paper_id",
                                                  paper_vector_col=paper_output_column,
                                                  peer_paper_vector_col="peer_paper_tf_idf_vector",
                                                  output_col=self.features_col, label_col="label")


        dataset = papersPairBuilder.transform(dataset)
        lsvc = LinearSVC(maxIter=10, regParam=0.1, featuresCol=self.features_col,
                         labelCol="label")
        # Fit the model
        lsvcModel = lsvc.fit(dataset)
        self.model = lsvcModel
        return self.model

    def _transform(self, dataset):
        """
        Add prediction to each paper in the input data set based on the trained model and its features vector.
        
        :param dataset: paper profiles
        :return: dataset with predictions - column "prediction"
        """
        self.paper_profiles_model.setPaperIdCol(self.paperId_col)
        self.paper_profiles_model.setOutputCol(self.features_col)

        # add paper representaion to each paper in the corpus
        papers_corpus = self.paper_profiles_model.transform(dataset)

        # make predictions using the model over full papers corpus
        papers_corpus_predictions = self.model.transform(papers_corpus)

        return papers_corpus_predictions


class PeerPapersSampler(Transformer):
    """
    Generate peer papers for a paper. For each paper in the data set, a list of peer papers will be selected from the paper corpus.
    The peer papers are selected randomly.
    """

    def __init__(self, papers_corpus, peer_papers_count, paperId_col="paper_id", userId_col="user_id", output_col="peer_paper_id"):
        """
        :param papers_corpus: PaperCorpus object that contains data frame. It represents all papers from the corpus. 
        Possible format (paper_id, citeulike_paper_id). See PaperCorpus documentation for more information.
        :param peer_papers_count: number of peer papers that will be sampled per paper
        :param paperId_col name of the paper id column in the input data frame of transform()
        :param userId_coln name of the user id column in the input data frame of transform()
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
        indexed_papers_corpus = papers.withColumn('paper_id_index',
                                                  F.row_number().over(Window.orderBy(self.papers_corpus.paperId_col)))

        # add the generated index to each paper in the input dataset
        indexed_dataset = dataset.join(indexed_papers_corpus,
                                       indexed_papers_corpus[self.papers_corpus.paperId_col] == dataset[
                                           self.paperId_col]).drop(self.papers_corpus.paperId_col)

        total_papers_count = indexed_papers_corpus.count()

        # collect positive papers per user
        user_library = indexed_dataset.select(self.userId_col, "paper_id_index").groupBy(self.userId_col) \
            .agg(F.collect_list("paper_id_index").alias("indexed_positive_papers_per_user"))

        # add list of positive papers per user
        dataset = dataset.join(user_library, self.userId_col)

        # generate peer papers
        dataset = dataset.withColumn("indexed_peer_papers_per_user", UDFContainer.getInstance()
                                     .generate_peers_udf("indexed_positive_papers_per_user", F.lit(total_papers_count),
                                                         F.lit(self.peer_papers_count)))

        # drop columns that we have added
        dataset = dataset.drop("indexed_positive_papers_per_user")

        # explode peer paper ids column
        dataset = dataset.withColumn("indexed_peer_paper_id", F.explode("indexed_peer_papers_per_user")) \
            .drop("indexed_peer_papers_per_user")

        # rename the column before the join to not mix them
        dataset = dataset.withColumnRenamed(self.paperId_col, "positive_paper_id")

        # revert indexed_peer_paper_id to oridinal paper_id
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
    There are three options for pair building:
    1) duplicated_pairs - In the given example above, the difference between (p - p_p) will be computed and added with class 1.
    As well as (p_p - p) with class 0.
    2) one_class_pairs - only the difference between (p - p_p) will be computed and added with class 1.
    3) equally_distributed_pairs - because each paper p has a set of corresponding peer papers, for 50% of them (p - p_p) will 
    be computed and added wirh class 1. And for the other 50%, (p_p - p) with class 0.
    """

    def __init__(self, pairs_generation, paperId_col="paper_id", peer_paperId_col="peer_paper_id",
                 paper_vector_col="paper_vector", peer_paper_vector_col="peer_paper_vector",
                 output_col="pair_paper_difference", label_col="label"):
        """
        Constructs the builder.


        :param pairs_generation: there are three possible values: duplicated_pairs, one_class_pairs, 
        equally_distributed_pairs.For example, if we have a paper p, and a set of peer papers N for the paper p
        1) if it is "duplicated_pairs" - for each paper p_p of the set N, calculate (p - p_p, class:1) and 
        (p_p - p, class: 0)
        2) if it is "one_class_pairs" - for each paper p_p of the set N, calculate (p - p_p, class:1)
        3) if it is "equally_distributed_pairs" - for 50% of papers in the set N, calculate p - p_p, class:1), 
        and for the other 50% (p_p - p, class: 0)
        :param paperId_col name of the column that contains paper id
        :param peer_paperId_col name of the column that contains peer paper id
        :param paper_vector_col name of the column that contains representation of the paper
        :param peer_paper_vector_col name of the column that contains representation of the peer paper
        :param output_col: name of the column where the result vector is stored
        :param label_col name of the column where the class of the pair/result is stored
        """
        self.pairs_generation = pairs_generation
        self.paperId_col = paperId_col
        self.peer_paperId_col = peer_paperId_col
        self.paper_vector_col = paper_vector_col
        self.peer_paper_vector_col = peer_paper_vector_col
        self.output_col = output_col
        self.label_col = label_col


    def _transform(self, dataset):
        if(self.pairs_generation == "equally_distributed_pairs"):
            # 50 % of the paper_pairs with label 1, 50% with label 0

            # get a list of peer paper ids per paper
            peers_per_paper = dataset.groupBy(self.paperId_col).agg(F.collect_list(self.peer_paperId_col).alias("peers_per_paper"))

            # generate 50/50 distribution to positive/negative class
            peers_per_paper = peers_per_paper.withColumn("equally_distributed_papers", UDFContainer.getInstance().split_papers_udf("peers_per_paper"))

            # positive label 1
            positive_class_per_paper = peers_per_paper.withColumn("positive_class_papers", F.col("equally_distributed_papers")[0])

            positive_class_per_paper = positive_class_per_paper.select(self.paperId_col, F.explode("positive_class_papers").alias(self.peer_paperId_col))
            # fix this
            positive_class_dataset = dataset.join(positive_class_per_paper, (positive_class_per_paper[self.peer_paperId_col] == dataset[self.peer_paperId_col])
                                                                        & (positive_class_per_paper[self.paperId_col] == dataset[self.paperId_col]) )

            # add the difference (paper_vector - peer_paper_vector) with label 1
            positive_class_dataset = positive_class_dataset.withColumn(self.output_col, UDFContainer.getInstance().vector_diff_udf(
                                                            self.paper_vector_col, self.peer_paper_vector_col))
            # add label 1
            positive_class_dataset = positive_class_dataset.withColumn(self.label_col, F.lit(1))

            # negative label 0
            negative_class_per_paper = peers_per_paper.withColumn("negative_class_papers", F.col("equally_distributed_papers")[1])
            negative_class_per_paper = negative_class_per_paper.select(self.paperId_col, F.explode("negative_class_papers").alias(self.peer_paperId_col))
            negative_class_dataset = dataset.join(negative_class_per_paper, (negative_class_per_paper[self.peer_paperId_col] == dataset[self.peer_paperId_col]) \
                                                    & (negative_class_per_paper[self.paperId_col] == dataset[self.paperId_col]))

            # add the difference (peer_paper_vector - paper_vector) with label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.output_col, UDFContainer.getInstance().vector_diff_udf(self.peer_paper_vector_col,
                                                            self.paper_vector_col))
            # add label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.label_col, F.lit(0))
            dataset = positive_class_dataset.union(negative_class_dataset)
        elif(self.pairs_generation == "duplicated_pairs"):
            # add the difference (paper_vector - peer_paper_vector) with label 1
            positive_class_dataset = dataset.withColumn(self.output_col,
                                         UDFContainer.getInstance().vector_diff_udf(self.paper_vector_col, self.peer_paper_vector_col))
            # add label 1
            positive_class_dataset = positive_class_dataset.withColumn(self.label_col, F.lit(1))

            # add the difference (peer_paper_vector - paper_vector) with label 0
            negative_class_dataset = dataset.withColumn(self.output_col,
                                                        UDFContainer.getInstance().vector_diff_udf(self.peer_paper_vector_col,
                                                                        self.paper_vector_col))
            # add label 0
            negative_class_dataset = negative_class_dataset.withColumn(self.label_col, F.lit(0))

            dataset = positive_class_dataset.union(negative_class_dataset)
        elif(self.pairs_generation == "one_class_pairs"):
            # add the difference (paper_vector - peer_paper_vector) with label 1
            dataset = dataset.withColumn(self.output_col,
                                         UDFContainer.getInstance().vector_diff_udf(self.paper_vector_col, self.peer_paper_vector_col))
            # add label 1
            dataset = dataset.withColumn(self.label_col, F.lit(1))
        else:
            # throw an error - unsupported option
            raise ValueError('The option' + self.pairs_generation + ' is not supported.')
        return dataset