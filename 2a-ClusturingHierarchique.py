import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import fcluster



# Chargement des features
features = pd.read_csv("features_clients.csv")

# Sélection UNIQUEMENT des colonnes numériques pour le clustering
numeric_cols = ["mean_amount", "std_amount", "frequency", "n_virements", "n_retraits", "n_paiements"]
X = features[numeric_cols]
X = X.fillna(0)  # Remplacer les NaN par 0


# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Z = linkage(X_scaled, method='ward')

# VIsualisation du dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90., leaf_font_size=12.)
plt.title("Dendrogramme des clients")
plt.xlabel("Index des clients")
plt.ylabel("Distance de Ward")
plt.savefig("Dendogramme_clusters.png")
plt.show()


# Extraire clusters
max_clusters = 50
clusters = fcluster(Z, t=max_clusters, criterion='maxclust')

# Extraire les clusters en coupant le dendrogramme à une distance donnée
distance_threshold = 5# À adapter selon le dendrogramme
fclusters = fcluster(Z, t=distance_threshold, criterion='distance')


# Ajouter les labels de clusters  dans le DataFrame features
features['cluster'] = fclusters


# Sauvegarde des résultats
features.to_csv("clients_with_clusters.csv", index=False)
print("Résultats du clustering sauvegardés dans 'clients_with_clusters.csv'.")



numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
cols_to_group = numeric_cols.union(['cluster'])

cluster_summary = features[cols_to_group].groupby('cluster').mean()
print(cluster_summary)

# Clustering
# Sauvegarde des résultats
cluster_summary.to_csv("clients_with_clusters_Hiererachique.csv", index=False)
print("Résultats du clustering sauvegardés dans 'clients_with_clusters_Hiererachique.csv'.")