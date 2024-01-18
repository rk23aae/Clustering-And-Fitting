# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to handle non-numeric values and convert data to numeric format
def preprocess_data(data):
    return pd.to_numeric(data, errors='coerce').fillna(0).values

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Function to perform k-means clustering
def k_means_clustering(data, k, max_iterations=100):
    # Randomly initialize centroids
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]
    
    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        assignments = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # Update centroids based on the mean of assigned points
        for i in range(k):
            centroids[i] = np.mean(data[assignments == i], axis=0)
    
    return centroids, assignments

# Function to compare countries within and between clusters
def compare_countries(data, assignments, centroids):
    # Choose one country from each cluster as a representative
    representatives = [np.random.choice(np.where(assignments == i)[0]) for i in range(len(centroids))]
    
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    
    # Plot each country with a unique marker and color for its cluster
    for i in range(len(data)):
        cluster_index = assignments[i]
        marker = 'o' if i in representatives else 'x'
        color = f'C{cluster_index}'  # Use different colors for different clusters
        plt.scatter(data[i, 0], data[i, 1], marker=marker, color=color, label=f'Country {i + 1}')

    # Plot centroids
    for i, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], marker='*', color=f'C{i}', s=200, label=f'Centroid {i + 1}')

    plt.title('Mortality rate, under-5 (per 1,000 live births)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

# Specify file path and name
file_path = r'C:\DOCUMENT\BUSINESS DOC\january jobs\API_SH.DYN'
file_name = 'API_SH.DYN.MORT_DS2_en_csv_v2_6300138.csv'
full_file_path = f'{file_path}\\{file_name}'

# Read data from the CSV file
df = pd.read_csv(full_file_path, skiprows=4)

# Select relevant columns for clustering
columns_for_clustering = ['Country Name', '1990', '1991', '1992', '1993', '1994', '1995', '1996',
                           '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',
                           '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
                           '2019', '2020', '2021']

# Extract relevant columns as features and preprocess the data
features = np.column_stack([preprocess_data(df[col]) for col in columns_for_clustering[1:]])

# Set the number of clusters (k)
k = 3

# Perform k-means clustering
centroids, assignments = k_means_clustering(features, k)

# Compare countries within and between clusters and plot the results
compare_countries(features, assignments, centroids)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, max_iters=100, random_state=None):
    """
    Basic K-Means clustering implementation.
    
    Parameters:
    - X: Data matrix (numpy array)
    - k: Number of clusters
    - max_iters: Maximum number of iterations
    - random_state: Seed for reproducibility
    
    Returns:
    - centroids: Final cluster centers
    - cluster_assignments: Index of the cluster each data point belongs to
    """
    if random_state:
        np.random.seed(random_state)
    
    # Randomly initialize centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Assign each data point to the closest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)
        
        # Update centroids based on mean of assigned data points
        centroids = np.array([X[cluster_assignments == i].mean(axis=0) for i in range(k)])
    
    return centroids, cluster_assignments

# Load your CSV file into a DataFrame
file_path = r'C:\DOCUMENT\BUSINESS DOC\january jobs\API_SH.DYN'
file_name = 'API_SH.DYN.MORT_DS2_en_csv_v2_6300138.csv'
full_file_path = f'{file_path}\\{file_name}'

# Assuming your DataFrame is named df
df = pd.read_csv(full_file_path, skiprows=4)  # Skip metadata rows

# Select relevant columns for clustering
columns_for_clustering = ['Country Name', 'Country Code', '1990', '1991', '1992', '1993', '1994', '1995', '1996',
                           '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',
                           '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
                           '2019', '2020', '2021']

# Extract the selected columns and rows, excluding non-numeric values in 'Country Name'
numeric_data = df[df['Country Name'].str.isnumeric() == False][columns_for_clustering]

# Drop rows with missing values
numeric_data = numeric_data.dropna()

# Choose the number of clusters (less than or equal to the number of data points)
num_clusters = min(3, numeric_data.shape[0])

# Convert DataFrame to numpy array
data_array = numeric_data.iloc[:, 2:].to_numpy()

# Perform K-Means clustering
centroids, cluster_assignments = kmeans(data_array, num_clusters, max_iters=100, random_state=42)

# Add cluster labels to the DataFrame
numeric_data['Cluster'] = cluster_assignments + 1  # Adding 1 because cluster labels start from 1 in K-Means

# Print cluster centers (in original form)
print("Cluster Centers:")
cluster_centers_df = pd.DataFrame(centroids, columns=['1990', '1991', '1992', '1993', '1994', '1995', '1996',
                                                     '1997', '1998', '1999', '2000', '2001', '2002', '2003',
                                                     '2004', '2005', '2006', '2007', '2008', '2009', '2010',
                                                     '2011', '2012', '2013', '2014', '2015', '2016', '2017',
                                                     '2018', '2019', '2020', '2021'])

print(cluster_centers_df)



# Visualize the results with enhanced styling and jitter
plt.figure(figsize=(12, 8))

# Define colors for each cluster
colors = ['blue', 'green', 'orange']

# Scatter plot of two features with jitter
for cluster_label in range(1, num_clusters + 1):
    cluster_data = numeric_data[numeric_data['Cluster'] == cluster_label]
    jitter = np.random.normal(scale=0.1, size=len(cluster_data))  # Adjust the scale as needed
    plt.scatter(cluster_data['1990'] + jitter, cluster_data['1991'] + jitter, c=colors[cluster_label - 1], label=f'Cluster {cluster_label}', s=70, alpha=0.7)

# Plot cluster centers
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

plt.xlabel('No of children ')
plt.ylabel('Mortality rate')
plt.title('Clustering Results (K-Means)')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t

# Define a logistic growth function
def logistic_growth(t, L, k, t0, A):
    return L / (1 + np.exp(-k * (t - t0))) + A

# Dummy data for demonstration
time_points = np.arange(1990, 2020)
observed_values = np.random.rand(len(time_points)) * 50  # Replace this with your actual data

# Fit the logistic growth model to the data
params, covariance = curve_fit(logistic_growth, time_points, observed_values, bounds=([0, 0, 0, 0], [100, 1, 2025, 100]))

# Rest of the code remains the same
# ...

# Plot the data, best-fit curve, and confidence intervals
plt.figure(figsize=(10, 6))
plt.scatter(time_points, observed_values, label='Observed Data', color='blue')
plt.plot(t_predictions, logistic_growth(t_predictions, *params), label='Best-Fit Curve', color='red')
plt.fill_between(t_predictions, lower_bound, upper_bound, color='pink', alpha=0.3, label='Confidence Interval (95%)')

plt.xlabel('Year')
plt.ylabel('Values')
plt.title('Logistic Growth Model Fitting with Confidence Intervals')
plt.legend()
plt.show()

