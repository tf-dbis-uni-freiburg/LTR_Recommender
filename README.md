
# Learning to Rank in a CBF Recommender, a trade-off between personalization and efficiency Master Thesis Proposal 


<p align="center">  Master thesis  <br/>
    Student: Polina Koleva <br/>
    Supervised by: Anas Alzogbi

## Introduction  
  The number of research papers has increased tremendously over the past decade. Consequently, searching for papers of interest has become really time-consuming task. While a rich amount of related work tackling the problem is available, we restrict our focus on content-based scenarios considering only information from the active user. More precisely, the paper [1] that proposes a new approach for solving the problem can be considered as our starting point. In it, the personalized paper recommendation system is formalized as a ranking problem with respect to users’ interest. Initially, a set of relevant papers for each user is provided. To extend it, a way for finding irrelevant papers to a user is proposed. Then a supervised pairwise learning to rank approach is employed to predict recommendations.  <br /> 
  The usage of learning-to-rank has been proven to result in success in many areas, especially in Information Retrieval [2]. The most famous case in which such algorithms are used is in search engines.In them, a prediction model is trained based on information about queries and their relevant documents. Afterwards, the model is able to rank the most similar documents to a given query. Providing a ranked list of recommendations has a lot of similarities with ranking a list of documents matching to a query. Moreover, the learning-to-rank techniques show good performance in practice [2]. The mentioned reasons validate why the proposed method [1] outperforms existing heuristics- and model-based algorithms for recommendations. But although its achievements, we found some issues which need consideration.  <br /> 
  Firstly, a ranking model is trained per user basis. The time complexity of the training phase increases proportionally to the number of users. If it is large, training so many models could be really expensive. Taking into account the fast growing amount of data and the need of immediate result of a recommender system, such an obstacle could make the method inapplicable in practice. <br /> 
  Secondly, the approach is verified only by using one pairwise learning-to-rank algorithm. We believe that it is beneficial to explore more than one particular algorithm, so that a valid and stable results can be provided.  

## Motivation
  The goal of a ranking system is to find the best possible ordering of a set of items for a user, within a specific context, in real-time. If we follow the proposed approach in the paper [1], to train a model per user can be time-consuming and expensive. This leads to slow response time, which make the method inapplicable in practice. For this reason, we have to find is there a better way to do it? We are interested in keeping the accuracy high so that relevant papers to a user are recommended, but also improving the response time. There are multiple theories that have to be explored.<br /> 
  On one hand, we can keep the “one model per user” suggestion. We can argue that when we train a model per user, the recommendations are personalized. Therefore, we might try to find a way of parallelize the training phase. This way, we can gain from a distributed computations, so that we will preserve the accuracy high but we will reduce the time complexity of training. <br /> 
  On the other hand, we can borrow a working theory already proven in Information Retrieval context. In reality, the input space for search engines consists of set of queries and documents. For each query, there is a set of relevant documents. When a learning-to-rank approach is used, the ranking model is trained over the whole set of data. As a result, only one general ranking model is produced which is applied for all new coming queries. If we match a query in search engine context to a user in recommender system context, we can use the same strategy. If we model our input space as set of all users and their relevant papers, we can come up with one ranking model. Therefore, the training time will be reduced but as well as the accuracy. This suggestion seems not enough personalized to an user’s interests which is an important functionality of a recommender system.<br /> 
  To summarize, on one side we have a general model with less accuracy, but not so expensive in terms of training. While on the other hand, we can train a model per user which will be more personal to user preferences, but the training process is time-consuming.  So after a comparison between two general approaches for generating a ranking model, a trade off might be made in terms of accuracy versus complexity/efficiency and a new approach in between might be considered. The question here is can we come up with something in between by using optimization techniques?  

## Proposed contribution
1. Study different pairwise learning to rank algorithms (RankingSVM, LambdaRank) to solve the scientific paper recommendations. [3] [4] [5]
2. Study the available ways of distributions and parallel execution of those algorithms.
3. Investigate in different approaches and study the influence of building a single ranking model vs. one ranking model per each user. Make a comparison between the two approaches. 
4. Consider constructing a model “in between”. In order to build such a model, we can investigate in already known ways used in Information Retrieval. For example, clustering of users and personalization of a general ranking.
5. Provide an efficient implementation exploiting distributed frameworks (Spark)
6. Investigate in feature extraction of documents. Papers are represented in terms of feature vectors. The accuracy of a prediction model is based on the relevance of these features. Therefore, it will be beneficial to add new features that will improve the built model. [6]

## References  
[1] A. Alzoghbi, V. A. A. Ayala, P. M. Fischer, and G. Lausen. Learning-to-Rank in research paper CBF recommendation: Leveraging irrelevant papers. 2016  
[2] Tie-Yan Liu. Learning to Rank for Information Retrieval. 2011 Chih-Wei Hsu, Chih-Chung Chang, and Chih-Jen Lin. A Practical Guide to Support Vector Classification. 2016  
[3]  T. Joachims. Optimizing Search Engines Using Clickthrough Data. Proceedings of the ACM Conference on Knowledge Discovery and Data Mining (KDD), ACM. 2002.  
[4] Christopher J.C. Burges. From RankNet to LambdaRank to LambdaMART: An Overview. 2010 
[5] Huan Xue, Jiafeng Guo, Yanyan Lan and Lei Cao. Personalized paper Recommendation in Online Social Scholar System. 2014


## How to run

```
spark2-submit main.py -input <folder where input data is> -model_training <gm| imps | imp | cm> -peers_count <number> -pairs_generation <dp | ocp | edp>
```
