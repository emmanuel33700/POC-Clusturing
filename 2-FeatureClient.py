import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("operations_bancaires.csv")

# Conversion de la date
df["date"] = pd.to_datetime(df["date"])

# Agrégation par clients
features = df.groupby("client_id").agg(
    total_amount=("amount", "sum"),
    mean_amount=("amount", "mean"),
    std_amount=("amount", "std"),
    n_operations=("amount", "count"),
    n_virements=("operation_type", lambda x: (x == "virement").sum()),
    n_retraits=("operation_type", lambda x: (x == "retrait").sum()),
    n_paiements=("operation_type", lambda x: (x == "paiement").sum()),
    first_date=("date", "min"),
    last_date=("date", "max"),
).reset_index()

# Calcul de la fréquence moyenne (en jours)
features["frequency"] = features["n_operations"] / ((features["last_date"] - features["first_date"]).dt.days + 1)

# Sélection des features pour le clustering
X = features[["mean_amount", "std_amount", "frequency", "n_virements", "n_retraits", "n_paiements"]]
X = X.fillna(0)  # Remplacer les NaN par 0

# Normalisation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sauvegarde des features
features.to_csv("features_clients.csv", index=False)
print("Features calculées et sauvegardées dans 'features_clients.csv'.")
