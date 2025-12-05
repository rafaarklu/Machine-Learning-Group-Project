import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
import pandas as pd

# ------------------------------
# 1) Ler a base
# ------------------------------
df = pd.read_csv('base_Projeto_integrado.csv')

# Transformar 'combustivel' em número

# Selecionar duas features numéricas para clusterização
X = df[["price", "km"]].values

plt.figure(figsize=(12, 10))

# ------------------------------
# 2) Executar K-Means
# ------------------------------
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X)

# ------------------------------
# 3) Plot dos clusters
# ------------------------------
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=70)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='*', s=250, label='Centroids')

plt.title('K-Means Clustering - Base Projeto Integrado')
plt.xlabel('Combustível (codificado)')
plt.ylabel('Preço')
plt.legend()

# Salvar como SVG
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
