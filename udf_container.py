import datetime

class UDFContainer():

        # generate k negative papers
        def buildPublicationDate(year, input_month):
            if(input_month == None):
                input_month = "jan"
            # convert month name to month number
            print(input_month)
            month_number = datetime.datetime.strptime(input_month, '%b').month
            # always from the first of the month
            row_date = datetime.datetime(int(year), month_number, 1)
            return row_date

        # build_publication_date_udf = F.udf(buildPublicationDate, TimestampType())
        # papers_corpus = valid_year_papers.withColumn("pub_date", build_publication_date_udf("year", "month"))


