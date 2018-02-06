from pyspark.ml.base import Estimator
from pyspark.ml.base import Transformer
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT

# TODO add comments
class TFVectorizer(Estimator):

    # TODO possible parameters that we can add minDF, minTF

    term_corpus = []
    voc_size = 0
    mapping = {}
    # TODO if needed we can store the terms not only their ids
    # vocabulary = []

    def __init__(self, spark, papers_corpus):
        self.spark = spark
        self.papers_corpus = papers_corpus

    """
    Dataframe has a format "(paper_id, terms_count, term_occurrence)" 
    """
    def _fit(self, papers):

        """
        TODO rewrite comments
        Transforms the input dataset.

        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: transformed dataset
        """
        # select only those papers that are part of the paper corpus
        papers = papers.join(self.papers_corpus, "paper_id")

        # collect all distinct term ids
        terms = papers.select(F.explode("term_occurrence")).select("key").distinct()
        self.term_corpus = [int(i.key) for i in terms.collect()]
        # corpus size 224029
        self.voc_size = len(self.term_corpus)

        # create mapping to each term map a id, sequential
        #  0-index based
        for i in range(len(self.term_corpus)):
            self.mapping[self.term_corpus[i]] = i

        # broadcast the mapping
        broadcasted_mapping = self.spark.sparkContext.broadcast(self.mapping)

        print(self.voc_size)
        # mapping - map key - term_id, value - index in the vector
        # map - terms representation of a paper
        def map_to_vector(map, voc_size):
            vector_map = {}
            for key in map:
                index = broadcasted_mapping.value[int(key)]
                vector_map[index] = map[key]
            return Vectors.sparse(voc_size, vector_map)

        # register the udf
        mapToVectorUDF = udf(map_to_vector, VectorUDT())

        dataset = papers.withColumn("tf_vector", mapToVectorUDF("term_occurrence", F.lit(self.voc_size)))

        paper_profiles = dataset.select("paper_id", "tf_vector")
        return TFVectorizorModel(paper_profiles);


# TODO add comments
class TFVectorizorModel(Transformer):

    def __init__(self, paper_profiles,):
        # format (paper_id, tf_vector)
        self.paper_profiles = paper_profiles

    # add for each paper, its corresponding tf vectors
    # input column for paper id = "paper_id"
    def _transform(self, dataset):
        dataset = dataset.join(self.paper_profiles, "paper_id");
        return dataset
