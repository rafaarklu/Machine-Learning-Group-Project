import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

plt.figure(figsize=(12, 10))

# ==========================
# CARREGAR BASE
# ==========================
df = pd.read_csv('base_Projeto_integrado.csv')

# ==========================
# PRÉ-PROCESSAMENTO
# ==========================
df = df.dropna(subset=['cambio', 'km', 'price']) 

# codificar câmbio (classe a ser prevista)
encoder = LabelEncoder()
df['cambio_encoded'] = encoder.fit_transform(df['cambio'])

# Selecionar apenas 2 features para manter o estilo do código original
X = df[['km', 'price']].values
y = df['cambio_encoded'].values

# padronizar (essencial para KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================
# TREINAR O MODELO KNN
# ==========================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# ==========================
# VISUALIZAÇÃO DA FRONTEIRA DE DECISÃO
# ==========================
h = 0.02  # step do grid

x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)

sns.scatterplot(
    x=X_scaled[:, 0], y=X_scaled[:, 1],
    hue=df['cambio'], style=df['cambio'],
    palette="deep", s=100
)

plt.xlabel("km (padronizado)")
plt.ylabel("price (padronizado)")
plt.title("KNN Decision Boundary — Previsão de câmbio (k=3)")

# Exportar para SVG como no exemplo
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
