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
- Jun-Sep, MarMar, {, FebMar, JanAug-JanSep, Dec--JanMay\~{, FebFeb--FebJun, FebFeb, Jun--FebAug\~{}

### Data size(terms_keywords_based): in total 2.13 GB
- current 1.87 GB
- papers.csv 163.7 MB
- mult.dat 67.8 MB
- citeulikeId_docId_map.dat 2.4 MB
- citeulikeUserHash_userId_map.dat 1.1 MB

### Loading and Splitting statistics
- Local Mode: Splitting the data into folds and storing all 23 folds takes around 10 hours
- Local Mode: Loading folds and running statistics over then around 20 mins
- Local Mode: Training a singe SVM model for all users takes 56 mins over the first fold
- Local Mode: Training a singe SVM model for one user ()

- Cluster mode: Splitting the data into folds and storing all 23 folds takes around 1.2 hour
- Cluster mode: Training a singe SVM model for all users takes
- Cluster mode: Training a singe SVM model for one user ()


### New data set (10.05.2018)
Creating and storing folds (LOCAL MODE) takes: 8 mins
In total 5 folds: Data in period [2004-11-04, 2007-12-31].
 - 1 fold - Training data [2004-11-04, 2005-05-04], Test data [2005-05-04, 2005-11-04]
 - 2 fold - Training data [2004-11-04, 2005-11-04], Test data [2005-11-04, 2006-05-04]
 - 3 fold - Training data [2004-11-04, 2006-05-04], Test data [2006-05-04, 2006-11-04]
 - 4 fold - Training data [2004-11-04, 2006-11-04], Test data [2006-11-04, 2007-05-04]
 - 5 fold - Training data [2004-11-04, 2007-05-04], Test data [2007-05-04, 2007-11-04]
 
 > Statistics for the folds can be found in new-dataset-folds-statistics.txt file.
 
 #### Fold 1
 - paper corpus for fold 1: 159 453 papers
 - number of ratings in the training set 10333
 - After LDA transfrom number of ratings -> 10297 ???
 - after pair generation 10333 
 
