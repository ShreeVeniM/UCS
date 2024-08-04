**PROJECT OVERVIEW**


**Introduction**
This project aims to perform customer segmentation analysis using clustering techniques on a dataset of mall customers. The analysis includes visualizations and evaluations of different clustering methods to determine the optimal number of clusters.

**Directory Structure**
.git: Contains Git version control data.
charts: Directory for storing generated charts and visualizations.
main.py: The main script that orchestrates data loading, clustering, and visualization.
readme.md: Documentation file for the project.
requirements.txt: Lists the Python dependencies required for the project.
src: Source code directory containing modules for data loading, visualization, and clustering.
Key Components
main.py
The central script that coordinates the following tasks:

**Data Loading:**
Loads the customer data from src/dataset/mall_customers.csv.

**Visualization:**
Creates a pairplot of the dataset.
Generates scatter plots to visualize clusters.
Plots the Elbow method and Silhouette scores to evaluate clustering performance.

**Clustering:**
Performs KMeans clustering on the dataset with different feature combinations and cluster counts.
Calculates Within-Cluster Sum of Squares (WSS) for the Elbow method.
Calculates Silhouette scores for evaluating cluster quality.

requirements.txt
Specifies the Python libraries required for the project:

pandas
numpy
seaborn
matplotlib
scikit-learn

**Conclusion**
This project provides a comprehensive analysis of mall customer data using clustering techniques. It includes various visualizations to aid in understanding the clustering results and determining the optimal number of clusters.
