# Team19_Assignment3

Team member: Dada Zhang, Natalia Trejo

This python script is used for performing k-means algorithm. Download 'kmeans_algorithm.py' file and iris.arff data to implement this program.

Dataset: 
- iris.arff (sourced from Kaggle: https://www.kaggle.com/datasets/darshanchopra2241314/iris-arff-file)
- It is popular and well known flower dataset for classification

Implementation:
- For example: !python kmeans_algorithm.py iris.arff 3 0.0001 100
- where iris.arff is dataset, 3 is number of clusters, 0.0001 is epsilon, 100 is number of iterations.

The basic command is 
- !python kmeans_algorithm.py [iris.arff] [k] [epsilon] [iterations]


Results:
- Final cluster centroids, final SSD, total number of iterations
- Plot of runtime vs. number of clusters
- Plot of runtime vs. number of dimension
- Plot of runtime vs. size of dataset (number of transactions)

WeKa_Output:
- The results of k-means are saved in Weka_Output folder.
- Number of clusters is from 1 to 10

