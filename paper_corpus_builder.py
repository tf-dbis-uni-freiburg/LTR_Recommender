class PaperCorpusBuilder():
    """
    Class responsible for building the paper corpus. It consists of all papers (their citeulike_paper_ids) that
    will be considered in the next stages of the algorithm. For example, when all terms in the corpus are extracted,
    only terms from papers part of the paper corpus will be taken into account.
    """

    def buildCorpus(self, papers, end_year):
        """
        Extract all papers which are published before a particular year. These papers are considered as paper corpus for
        all next stages of the algorithm. 
        NOTE: Invalid values for paper year are null and -1. All papers that have such values are included in the paper corpus.
        
        :param papers: dataframe of all papers. Format -> 
            (citeulike_paper_id, type, journal, book_title, series, publisher, pages, volume, number, year, month, 
            postedat, address, title, abstract)
        :param end_year: all papers published before this date are selected 
        :return: dataframe that contains paper ids from all papers in the corpus
        """
        # filter all papers which have "year" equals to null
        null_year_papers = papers.filter(papers.year.isNull())

        # filter by end_year
        papers_corpus = papers.filter(papers.year <= end_year)

        # add papers with null year
        papers_corpus = papers_corpus.union(null_year_papers)

        # drop all columns that won't be used anymore
        papers_corpus = papers_corpus.select("citeulike_paper_id")
        return papers_corpus
