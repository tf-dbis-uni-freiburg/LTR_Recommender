class PaperCorpusBuilder():
    """
    Class responsible for building the paper corpus. It consists of all papers (their citeulike_paper_ids) that
    will be considered in the next stages of the algorithm. For example, when all terms in the corpus are extracted,
    only terms from papers part of the paper corpus will be taken into account.
    """
    @staticmethod
    def buildCorpus(papers, papers_mapping, end_year, paperId_col="paper_id", citeulikePaperId_col="citeulike_paper_id"):
        """
        Extract all papers which are published before a particular year. These papers are considered as paper corpus for
        all next stages of the algorithm. Each paper in the papers data frame is mapped by citeulike paper id. But this
        is the only data frame in the data set which use this type of id. Therefore, a mapping between citeulike paper ids
        and paper ids used in the other files in the data set is used, so that to each paper, its paper id is added.
   
        NOTE: Invalid values for paper year are null and -1. All papers that have such values are included in the paper 
        corpus.
        
        :param papers: dataframe of all papers. Format -> (citeulike_paper_id, type, journal, book_title, series, publisher, 
        pages, volume, number, year, month, postedat, address, title, abstract)
        :param papers_mapping data frame of mapping between paper_id and citeulike_paper_id
        :param paperId_col name of the paper id column in the papers_mapping dataframe
        :param citeulikePaperId_col name of the paper id column in the papers dataframe
        :param end_year: all papers published before this date are selected 
        :return: data frame that contains paper ids from all papers in the corpus. It consists of 2 columns with names 
        same as paperId_col and citeulikePaperId_col
        """
        # filter all papers which have "year" equals to null
        null_year_papers = papers.filter(papers.year.isNull())

        # filter by end_year
        papers_corpus = papers.filter(papers.year <= end_year)

        # add papers with null year
        papers_corpus = papers_corpus.union(null_year_papers)

        # drop all columns that won't be used anymore
        papers_corpus = papers_corpus.select(citeulikePaperId_col)

        # add paper_id to the corpus
        papers_corpus = papers_corpus.join(papers_mapping, citeulikePaperId_col)
        papers_corpus = PapersCorpus(papers_corpus, paperId_col, citeulikePaperId_col)
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

