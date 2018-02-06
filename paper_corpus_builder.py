"""
TODO explain what is paper corpus
TODO give example how the corpus is used in different steps from the algorithm pipeline
"""
class PaperCorpusBuilder():

    def buildCorpus(self, papers, end_year):
        """
        Extract all papers which are published before a particular year. These papers are considered as paper corpus for
        all next stages of the algorithm. 
        
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
