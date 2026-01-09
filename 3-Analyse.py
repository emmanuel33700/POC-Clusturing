import pandas as pd

# Chargement des données avec clusters
features = pd.read_csv("clients_with_clusters.csv")

# Colonnes numériques à analyser
numeric_cols = ["mean_amount", "std_amount", "frequency", "n_virements", "n_retraits", "n_paiements"]

# Moyenne par cluster
cluster_summary = features.groupby("cluster")[numeric_cols].mean()

# Affichage
print(cluster_summary)

# Sauvegarde pour analyse ultérieure
cluster_summary.to_csv("cluster_summary.csv")

################################
# visulaisation des lucters PCAs
#################################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Réduction de dimension
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features[numeric_cols])

# DataFrame pour visualisation
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["cluster"] = features["cluster"]

# Visualisation
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", palette="viridis", alpha=0.7)
plt.title("Projection PCA des clusters de clients")
plt.savefig("pca_clusters.png")
plt.show()

##################################
# Visualisation des clusters TSNE
#################################


from sklearn.manifold import TSNE

# Réduction de dimension avec t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(features[numeric_cols])

# DataFrame pour visualisation
tsne_df = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
tsne_df["cluster"] = features["cluster"]

# Visualisation
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue="cluster", palette="viridis", alpha=0.7)
plt.title("Projection t-SNE des clusters de clients")
plt.savefig("tsne_clusters.png")
plt.show()


######################################
# Visualisation des caractéristique pas cluster (VIa boxplot)
######################################


# Boxplot par feature et par cluster
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=features, x="cluster", y=col)
    plt.title(f"Distribution de {col} par cluster")
plt.tight_layout()
plt.savefig("boxplot_by_cluster.png")
plt.show()



######################################
# Visualisation des caractéristique pas cluster (VIa Heatmaps)
######################################
# Normalisation pour la heatmap
cluster_norm = (cluster_summary - cluster_summary.mean()) / cluster_summary.std()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_norm, annot=True, cmap="coolwarm", center=0)
plt.title("Caractéristiques normalisées par cluster")
plt.savefig("heatmap_clusters.png")
plt.show()
