## 29.01.2018

1. Split reduces - #users, #papers, #terms
2. Sampling reduces - #papers

## Papers statistics
- 172079 total number of papers
- 19 with "null" year
- 4839 with "-1" year
- 53067 with "null" month
- 4682 with "-1" year and "null" month
- 6 with "null" year and "null" month

## Term statistics
- 224029 terms in total

## Found invalid values for "month" in papers.csv
- Jun-Sep, MarMar, {, FebMar, JanAug-JanSep, Dec--JanMay\~{, FebFeb--FebJun, FebFeb, Jun--FebAug\~{

# Local Mode: Splitting the data into folds and storing all 23 folds takes around 10 hours
# Local Mode: Loading folds and running statistics over then around 20 mins
# Local Mode: Training a singe SVM model for all users takes 56 mins over the first fold
# Local Mode: Training a singe SVM model for one user ()

# Data size(terms_keywords_based): in total 2.13 GB
- current 1.87 GB
- papers.csv 163.7 MB
- mult.dat 67.8 MB
- citeulikeId_docId_map.dat 2.4 MB
- citeulikeUserHash_userId_map.dat 1.1 MB

# Creating and storing folds on the cluster - 1.2h