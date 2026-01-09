import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Paramètres
np.random.seed(42)
random.seed(42)
n_clients = 1000
n_operations = 100000
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

# Types d'opérations
operation_types = ["virement", "retrait", "paiement", "depot", "prelevement"]

# Génération des clients
clients = [f"CLI_{i:04d}" for i in range(1, n_clients + 1)]

# Génération des opérations
data = []
for _ in range(n_operations):
    client = random.choice(clients)
    date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    amount = round(random.expovariate(1/500) * 100, 2)  # Distribution exponentielle pour les montants
    op_type = random.choices(operation_types, weights=[0.3, 0.2, 0.3, 0.1, 0.1])[0]
    data.append([client, date.strftime("%Y-%m-%d"), amount, op_type])

# Création du DataFrame
df = pd.DataFrame(data, columns=["client_id", "date", "amount", "operation_type"])

# Sauvegarde
df.to_csv("operations_bancaires.csv", index=False)
print("Jeu de données généré et sauvegardé dans 'operations_bancaires.csv'.")
