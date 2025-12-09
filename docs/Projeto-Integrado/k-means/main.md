# K-Means (Clusterização)

## O que é o K-Means?

O K-Means é um algoritmo **não supervisionado** utilizado para agrupar dados com características semelhantes. Ele divide os dados em **k grupos**, baseando-se na proximidade dos pontos em relação aos **centroides**.

### Como funciona:

1. Escolhe k centroides.
2. Atribui cada ponto ao centro mais próximo.
3. Recalcula os centroides.
4. Repete até não haver mudanças significativas.

### Utilidade:

* Detecção de padrões.
* Segmentação.
* Análises exploratórias.

---

## Exemplo de Execução (Agrupamento de Veículos)



```python exec="on" html="1"
--8<-- "docs\Projeto-Integrado\k-means\kmeans.py"
```



O exemplo abaixo agrupa os carros com base em duas variáveis numéricas: **preço** e **km**.

### Código Base Utilizado

```python
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv('base_Projeto_integrado.csv')
X = df[["price","km"]].values

plt.figure(figsize=(12, 10))

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=70)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='*', s=250, label='Centroids')

plt.title('K-Means Clustering - Base Projeto Integrado')
plt.xlabel('Preço')
plt.ylabel('KM')
plt.legend()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()
```

### Interpretação do Exemplo

* Os pontos foram agrupados em 3 clusters com base na proximidade entre **preço** e **km**.
* Cada cor representa um grupo formado pelo algoritmo.
* As estrelas vermelhas são os **centroides**: pontos que representam a "média" de cada grupo.

---

**Conclusão**

O exemplo aplicado com K-Means agrupa veículos com base em `price` e `km`, resultando em clusters que refletem segmentos com características semelhantes (por exemplo: veículos baratos com alta quilometragem, veículos caros com baixa quilometragem, etc.). Os centroides (marcados como estrelas vermelhas) representam a posição média de cada grupo e ajudam a interpretar o perfil típico de cada cluster.

Pontos importantes na interpretação:

- **Tamanho e densidade dos clusters**: clusters muito pequenos ou esparsos podem indicar outliers ou agrupamentos pouco significativos.
- **Separabilidade**: se os clusters se sobrepõem muito no gráfico, pode ser um sinal de que `k` escolhido não é ideal ou que as features não capturam bem a estrutura dos dados.
- **Centroides**: dão uma referência média para `price` e `km`, úteis para rotular segmentos (ex.: "baixo preço / alto km").

