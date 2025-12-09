import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
import numpy as np

plt.figure(figsize=(12, 10))



# Carregar dados do CSV do link
url = 'https://raw.githubusercontent.com/rafaarklu/Machine-Learning-Group-Project/refs/heads/main/Salary_Data.csv'
df = pd.read_csv(url)

# Verificar informações básicas dos dados
# print("Informações sobre o dataset:")
# print(df.info())
# print("\nPrimeiras linhas:")
# print(df.head())

# Verificar valores ausentes
#print("\nValores ausentes por coluna:")
#print(df.isnull().sum())

# Selecionar duas colunas numéricas para o K-means
# Usando 'Years of Experience' e 'Salary'
columns_for_clustering = ["Years of Experience", "Salary"]

# Verificar valores ausentes
# print("\nValores ausentes por coluna:")
# print(df.isnull().sum())

# OPÇÃO 1: Remover linhas com valores NaN (método usado)
df_clean = df[columns_for_clustering].dropna()
# print(f"\nDados originais: {len(df)} linhas")
# print(f"Dados após remoção de NaN: {len(df_clean)} linhas")

# OPÇÃO 2: Alternativa - Usar imputação para preencher valores NaN (comentado)
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy='median')  # ou 'mean', 'most_frequent'
# df_imputed = df[columns_for_clustering].copy()
# df_imputed[columns_for_clustering] = imputer.fit_transform(df_imputed[columns_for_clustering])
# df_clean = df_imputed

# Verificar se ainda temos dados suficientes
if len(df_clean) < 3:
    raise ValueError("Dados insuficientes após remoção de valores NaN")

X = df_clean.values

# Criar DataFrame limpo completo para análise posterior
df_for_analysis = df.dropna(subset=columns_for_clustering).copy()

# Normalização opcional para melhor clustering (comentado para manter escala original)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)


# Run K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X)

# Adicionar os rótulos ao DataFrame limpo
df_for_analysis['cluster'] = labels

# Mapear cores do cmap para clusters
import matplotlib as mpl
cmap = plt.get_cmap('viridis')
colors = [cmap(i / (kmeans.n_clusters - 1)) for i in range(kmeans.n_clusters)]

# Plotar cada cluster com legenda do nível de educação mais comum
for cluster in range(kmeans.n_clusters):
    cluster_data = df_for_analysis[df_for_analysis['cluster'] == cluster]
    # Nível de educação mais comum no cluster
    if not cluster_data.empty:
        common_education = cluster_data['Education Level'].mode()[0] if 'Education Level' in cluster_data.columns else f'Cluster {cluster}'
        plt.scatter(cluster_data['Years of Experience'], cluster_data['Salary'], 
                    color=colors[cluster], s=50, 
                    label=f'Cluster {cluster} ({common_education})')

# Plotar centróides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='*', s=200, label='Centroids')
plt.title('K-Means Clustering Results - Salary Analysis')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()

# Verificar a relação entre cluster e nível de educação (se a coluna existir)
# if 'Education Level' in df_for_analysis.columns:
#     print("Relação entre cluster e nível de educação:")
#     print(pd.crosstab(df_for_analysis['cluster'], df_for_analysis['Education Level']))

# Estatísticas por cluster
# print("\nEstatísticas por cluster:")
# for cluster in range(kmeans.n_clusters):
#     cluster_data = df_for_analysis[df_for_analysis['cluster'] == cluster]
#     print(f"\nCluster {cluster}:")
#     print(f"  Salário médio: ${cluster_data['Salary'].mean():.2f}")
#     print(f"  Anos de experiência médio: {cluster_data['Years of Experience'].mean():.1f}")
#     print(f"  Tamanho do cluster: {len(cluster_data)}")

# Print centroids and inertia
# print("\nCentróides finais:", kmeans.cluster_centers_)
# print("Inércia (WCSS):", kmeans.inertia_)

# # Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())