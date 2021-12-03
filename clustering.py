# The actual clustering of our 9-attribute data

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

print("***** Training Set *****")
train = pd.read_csv('mclarenAnomalous.data', sep=",", header=0)
print(train)
print("***** Training Set *****")

print("***** Clustering K-Means *****")
kmeans = KMeans(n_clusters=4)
X = np.array(train)
kmeans.fit(X)
# Cluster representation help obtained here: https://stackoverflow.com/questions/36195457/how-to-get-the-samples-in-each-cluster
cluster_map = pd.DataFrame()
cluster_map['data_index'] = train.index.values
cluster_map['cluster'] = kmeans.labels_
for i in range(5):
    print("*** Cluster " + str(i) + " ***")
    print(cluster_map[cluster_map.cluster == i])
print("***** Clustering K-Means *****")
