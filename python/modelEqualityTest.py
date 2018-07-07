import numpy
from random import shuffle
from fold_utils import FoldValidator
from learning_to_rank_spark2 import PeerPapersSampler, LearningToRank, PapersPairBuilder
from logger import Logger
import datetime

class EqualityTest():

    def __init__(self, spark, pairs_generation="edp", peer_papers_count="25", paperId_col="paper_id", userId_col="user_id"):
        self.spark = spark
        self.pairs_generation = pairs_generation
        self.peer_papers_count = peer_papers_count
        self.paperId_col = paperId_col
        self.userId_col = userId_col

    def IMPvsIMS(self):
        Logger.log("Equality test for IMP vs IMS...")
        start_time = datetime.datetime.now()
        for i in range(1, 6):
            fold = FoldValidator().load_fold(self.spark, i)

            # drop some unneeded columns
            # user_id | citeulike_paper_id | paper_id |
            fold.test_data_frame = fold.test_data_frame.drop("timestamp", "citeulike_user_hash")
            # citeulike_paper_id | user_id | paper_id|
            fold.training_data_frame = fold.training_data_frame.drop("timestamp", "citeulike_user_hash")

            # load peers so you can remove the randomization factor when comparing
            # 1) Peer papers sampling
            nps = PeerPapersSampler(fold.papers_corpus, self.peer_papers_count, paperId_col=self.paperId_col,
                                    userId_col=self.userId_col,
                                    output_col="peer_paper_id")

            # schema -> user_id | citeulike_paper_id | paper_id | peer_paper_id |
            peers_dataset = nps.load_peers(self.spark, i)

            # removes from training data frame those users which do not appear in the test set, no need
            # a model for them to be trained
            test_user_ids = fold.test_data_frame.select(self.userId_col).distinct()
            fold.training_data_frame = fold.training_data_frame.join(test_user_ids, self.userId_col)
            peers_dataset = peers_dataset.join(test_user_ids, self.userId_col)

            # test only for 10 random users
            user_ids = peers_dataset.select("user_id").distinct().collect()
            shuffle(user_ids)
            user_ids = user_ids[:50]
            Logger.log("Test for " + str(len(user_ids))+ " user/s")

            filter_condition = "user_id ==" + str(user_ids[0][0])
            for user_id in user_ids[1:]:
                filter_condition += " or user_id ==" + str(user_id[0])
            test_peer_paper = peers_dataset.filter(filter_condition)

            # peer_paper_lda_vector, paper_lda_vector
            papersPairBuilder = PapersPairBuilder(self.pairs_generation, "ims",
                                                      fold.ldaModel,
                                                      self.userId_col, self.paperId_col,
                                                      peer_paperId_col="peer_paper_id",
                                                      output_col="features",
                                                      label_col="label")
            # 2) pair building
            # paper_id | peer_paper_id | user_id | citeulike_paper_id | lda_vector | peer_paper_lda_vector | features | label
            dataset = papersPairBuilder.transform(test_peer_paper)
            dataset = dataset.drop("peer_paper_lda_vector", "lda_vector")
            dataset.persist()

            # Training LTR
            ltr_imp = LearningToRank(self.spark, fold.papers_corpus, fold.ldaModel, user_clusters=None,
                                     model_training="imp",
                                     pairs_generation=self.pairs_generation, peer_papers_count=self.peer_papers_count,
                                     paperId_col=self.paperId_col, userId_col=self.userId_col, features_col="features")

            imp_model = ltr_imp.fit(dataset)[0].modelWeights

            # Training LTR - sequential model
            ltr_ims = LearningToRank(self.spark, fold.papers_corpus, fold.ldaModel, user_clusters=None,
                                     model_training="ims",
                                     pairs_generation=self.pairs_generation, peer_papers_count=self.peer_papers_count,
                                     paperId_col=self.paperId_col, userId_col=self.userId_col, features_col="features")
            ims_model = ltr_ims.fit(dataset)

            # check equality in size
            assert len(ims_model) == len(ims_model)
            # check each weight vector
            for user_id, ims_weight in ims_model.items():
                imp_weight = imp_model[user_id]
                assert self.equals(imp_weight, ims_weight) == True
            Logger.log("Unpersist the data")
            dataset.unpersist()
            Logger.log("Equality test successfully completed for fold:" + str(i))

        end_time = datetime.datetime.now() - start_time
        Logger.log("Test finished in :" + str(end_time))
        Logger.log("Equality test completed.")

    def equals(self, v1, v2):
        equal = numpy.allclose(v1,v2)
        return equal