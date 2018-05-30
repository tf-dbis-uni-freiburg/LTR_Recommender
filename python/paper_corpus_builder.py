class PaperCorpusBuilder():
    """
    Class responsible for building the paper corpus. It consists of all papers (their paper_id and citeulike_paper_ids) that
    will be considered in the next stages of the algorithm. For example, when all terms in the corpus are extracted,
    only terms from papers part of the paper corpus will be taken into account. Paper corpus contains all papers part of a fold.
    """
    @staticmethod
    def buildCorpus(fold_papers, paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id"):
        """
        Paper corpus contains all papers in the fold (training and test set). Paper corpus is in
        format (paper_id, citeulike_paper_id)

        :param fold_papers all papers part of a fold. Format (paper_id, citeulike_papar_id)
        :param paperId_col name of the paper id column in the papers_mapping dataframe
        :param citeulikePaperId_col name of the paper id column in the papers dataframe
        :return: data frame that contains paper ids from all papers in the corpus. It consists of 2 columns with names 
        same as paperId_col and citeulikePaperId_col
        """
        # Filtering by year - not used anymore
        # NOTE: Invalid values for paper year are null and -1. All papers that have such values are included in the paper corpus.
        # filter all papers which have "year" equals to null
        # null_year_papers = papers.filter(papers.year.isNull())
        # filter by end_year
        # papers_corpus = papers.filter(papers.year <= end_year)
        # add papers with null year
        # papers_corpus = papers_corpus.union(null_year_papers)

        # add paper_id to the corpus
        papers_corpus = PapersCorpus(fold_papers, paperId_col, citeulikePaperId_col)
        return papers_corpus

class PapersCorpus:
    """
    Class that represents papers corpus. It contains a data frame of all papers in the corpus.
    For each paper, it is stored its paper id in the column with name @paperId_col.
    And its citeulike paper id in the column with name citeulikePaperId_col.
    """

    def __init__(self, papers, paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id"):
        self.papers = papers
        self.paperId_col = paperId_col
        self.citeulikePaperId_col = citeulikePaperId_col

