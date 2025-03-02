# Python script
import pandas as pd
import numpy as np
from scipy.io import arff
import argparse
import matplotlib.pyplot as plt
import time

# k-means algorithm function 
def kmeans(D, k, eps, iterations):
  # using 42 for random seed. necessary in the design.
  np.random.seed(42)
  D = np.array(D) 

  # Step 2: initial centroids (randomly)
  initial_centroids = np.random.choice(D.shape[0], k, replace=False)
  centroids = D[initial_centroids]

  # Step 2: repeating process (for loop)
  # initial list
  ssd_init = float('inf')    # initial
  ssd_list = np.array([])    # store
  labels = []

  # compute distance: centroids of each cluster
  for i in range(iterations):
    # Euclidean distance - d^2 = sum(x_ - c_)^2
    # euclidean_distance = np.sqrt(((D - centroids[:, np.newaxis])**2).sum(axis=2)) # using this, more modification for D
    euclidean_distance = np.linalg.norm(D[:, np.newaxis] - centroids, axis=2) #using this equation, same as above
    labels = np.argmin(euclidean_distance, axis=1)

    # Step 3: update centroids
    new_centroids = np.array([D[labels == j].mean(axis=0) if (labels == j).any() else centroids[j] for j in range(k)])

    # Step 4: compute SSD (aka SSE)
    # ssd = Sum(Sum(d^2))
    # ssd = sum(np.min(euclidean_distance, axis=1)**2)
    ssd = np.sum((D - centroids[labels]) ** 2)

    # Step 5: stopping program
    # If the change in the SSD < eps
    ssd_list = np.append(ssd_list, ssd)
    if ssd_list.size > 1:
      if abs(ssd_list[-1] - ssd_list[-2]) < eps:
        print(f"Stop program - SSD < epsilon, converged after {i+1} iterations")
        break

    ssd_init = ssd
    centroids = new_centroids
  # if the iterations is reached 
  if i == iterations - 1:
    print(f"Stop program - maximum iterations {iterations} reached out")

  return centroids, labels, ssd, i+1

#########################
# Plot 
#########################
# runtime of algorithm vs. number of clusters
# define k values
def plot_runtime_k(df, eps, iterations):
  k_values = range(1, 11)  # from 1 to 11. We will use same k values in WeKa
  runtimes = []            # same as Assignment 2
  
  for k in k_values:
    start_time = time.time()
    kmeans(df, k, eps, iterations)
    end_time = time.time()
    runtimes.append(end_time - start_time)
  
  # plot! #
  plt.figure(figsize=(8, 5))
  plt.plot(k_values, runtimes, marker='o', linestyle='-', color='r')
  plt.xlabel('Number of Clusters (k)')
  plt.ylabel('Runtime (seconds)')
  plt.title('Runtime of K-Means vs. Number of Clusters')
  plt.grid(True)
  plt.savefig('runtime_vs_clusters.png')
  plt.show()

# runtime of algorithm vs. number of dimensions
def plot_runtime_d(df, k, eps, iterations):
  dimensions = list(range(1, df.shape[1] + 1))
  runtimes = []

  for d in dimensions:
    start_time = time.time()
    kmeans(df.iloc[:, :d], k, eps, iterations)
    end_time = time.time()
    runtimes.append(end_time - start_time)

  # plot #
  plt.figure(figsize=(8, 5))
  plt.plot(dimensions, runtimes, marker='o', linestyle='-', color='b')
  plt.xlabel('Number of Dimensions (d)')
  plt.ylabel('Runtime (seconds)')
  plt.title('Runtime of K-Means vs. Number of Dimensions')
  plt.grid(True)
  plt.savefig('runtime_vs_dimension.png')
  plt.show()

# runtime of algorithm vs. size of dataset
def plot_runtime_n(df, k, eps, iterations):
  n_values = list(range(10, min(101, len(df)+1), 10))
  runtimes = []

  for n in n_values:
    start_time = time.time()
    # df_size is number of row
    # df_size = df.iloc[:n]
    df_size = df.sample(n=n, random_state=42)   # RSWR - random sample w/o replacement
    kmeans(df_size, k, eps, iterations)
    end_time = time.time()
    runtimes.append(end_time - start_time)

  # plot #
  plt.figure(figsize=(8, 5))
  plt.plot(n_values, runtimes, marker='o', linestyle='-', color='g')
  plt.xlabel('Number of Samples (n)')
  plt.ylabel('Runtime (seconds)')
  plt.title('Runtime of K-Means vs. Size of Dataset')
  plt.grid(True)
  plt.savefig('runtime_vs_size_of_data.png')
  plt.show()

# Goodness of clusters vs. number of clusters
def plot_goodness_k(df, eps, iterations):
  k_values = list(range(1, 11))
  goodness = []

  for k in k_values:
    centroids, labels, ssd, iters = kmeans(df, k, eps, iterations)
    goodness.append(ssd)

  # plot #
  plt.figure(figsize=(8, 5))
  plt.plot(k_values, goodness, marker='o', linestyle='-', color='purple')
  plt.xlabel('Number of Clusters (k)')
  plt.ylabel('Sum of Squares of Differences (SSD)')
  plt.title('Goodness of Clusters vs. Number of Clusters')
  plt.grid(True)
  plt.savefig('goodness_vs_clusters.png')
  plt.show()

# define main function - input parameter
def main():
  # define same arguments like assignment2
  parser = argparse.ArgumentParser(description="K-Means Algorithm")
  parser.add_argument("input_file", type=str, help="Path to the data ARFF file (.arff)")
  parser.add_argument("k", type=int, help="Number of clusters, k")
  parser.add_argument("eps", type=float, help="Epsilon value for stopping program (small value)")
  parser.add_argument("iterations", type=int, help="Total (maximum) number of iterations")

  args = parser.parse_args()

  # load data 
  data = arff.loadarff('iris.arff')
  df_raw = pd.DataFrame(data[0])
  # remove last column
  df = df_raw.iloc[:,:-1]     # keep the df contains continuous value

  # perform k-means algorithm: data, k, epsilon, iterations
  centroids, labels, ssd, iters = kmeans(df, args.k, args.eps, args.iterations)
  ## labels? no need for output.

  # The results have several components:
  # Final cluster centroids
  print("Final cluster centroids:\n", centroids)
  # Final SSD when stopping program
  print("Final SSD (stopping):", ssd)
  # Total Number of iterations when stopping 
  print("Total Number of Iterations:", iters)
  # we also want to save them is a text file ...
  with open("kmean_algorithm_results.txt", "w") as f:
    f.write(f"Final cluster centroids:\n{centroids}\n")
    f.write(f"Final SSD: {ssd}\n")
    f.write(f"Total Number of Iterations: {iters}\n")
  # add print to show it successed
  print("Ouput is stored in kmean_algorithm_results.txt")

  # show plots
  plot_runtime_k(df, args.eps, args.iterations)
  plot_runtime_d(df, args.k, args.eps, args.iterations)
  plot_runtime_n(df, args.k, args.eps, args.iterations)
  plot_goodness_k(df, args.eps, args.iterations)

if __name__ == "__main__":
  main()