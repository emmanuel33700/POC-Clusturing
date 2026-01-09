import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Chargement des features
features = pd.read_csv("features_clients.csv")

# Sélection UNIQUEMENT des colonnes numériques pour le clustering
numeric_cols = ["mean_amount", "std_amount", "frequency", "n_virements", "n_retraits", "n_paiements"]
X = features[numeric_cols]
X = X.fillna(0)  # Remplacer les NaN par 0

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=100, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
features["cluster"] = clusters

# Réduction de dimension pour la visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.6)
plt.title("Clusters de clients (PCA)")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.colorbar(label="Cluster")
plt.savefig("clusters_visualisation.png")
plt.show()

# Analyse descriptive des clusters
cluster_summary = features.groupby("cluster")[numeric_cols].mean()
print(cluster_summary)

# Sauvegarde des résultats
features.to_csv("clients_with_clusters.csv", index=False)
print("Résultats du clustering sauvegardés dans 'clients_with_clusters.csv'.")
