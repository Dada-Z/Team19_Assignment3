# Team19_Assignment3

Team member: Dada Zhang, Natalia Trejo

This python scriptp performs k-means algorithm. To implement the program, download the **kmeans_algorithm.py** file and **iris.arff** data.

Dataset: 
- iris.arff (sourced from Kaggle: https://www.kaggle.com/datasets/darshanchopra2241314/iris-arff-file)
- It is a popular and well-known flower dataset for classification

Implementation:
- !python kmeans_algorithm.py iris.arff 3 0.0001 100
- where iris.arff is a dataset, 3 is a number of clusters, 0.0001 is epsilon, and 100 is number of iterations.

The basic command is 
- !python kmeans_algorithm.py [iris.arff] [k] [epsilon] [iterations]


Results:
- Final cluster centroids, final SSD, total number of iterations (save in text file)
- Plot of runtime vs. number of clusters
- Plot of runtime vs. number of dimension
- Plot of runtime vs. size of dataset (number of transactions)
- Plot of goodness vs. number of clusters

WeKa_Output:
- The k-means results are saved in the Weka_Output folder.
- The number of clusters is from 1 to 10.

