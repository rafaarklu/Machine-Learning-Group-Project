# KNN — K-Nearest Neighbors

## O que é o KNN?

O KNN é um algoritmo supervisionado baseado em proximidade. Ele prevê a classe de um novo ponto olhando para suas **k vizinhos mais próximos**.

É amplamente utilizado em problemas de classificação.

### Características importantes:

* Baseado em distância (euclidiana na maioria dos casos).
* Necessita padronização dos dados.
* A decisão depende diretamente dos dados próximos.

### Intuição Geral

Se vários veículos semelhantes possuem o mesmo tipo de câmbio, um novo veículo com características parecidas provavelmente terá o mesmo câmbio.

---

## Exemplo de Execução (Classificação de Câmbio)


```python exec="on" html="1"
--8<-- "docs\Projeto-Integrado\knn\knn.py"
```



Aqui o KNN tenta prever o tipo de câmbio com base em **km** e **preço**, dois atributos que auxiliam na diferenciação do carro.

### Código Base Utilizado

```python
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
df = pd.read_csv('base_Projeto_integrado.csv')
df = df.dropna(subset=['cambio','km','price'])

encoder = LabelEncoder()
df['cambio_encoded'] = encoder.fit_transform(df['cambio'])

X = df[['km','price']].values
y = df['cambio_encoded'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# Fronteira de decisão
h = 0.02
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)

sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=df['cambio'], palette="deep", s=100)
plt.title("KNN Decision Boundary — Previsão de câmbio (k=3)")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()
```

### Interpretação do Exemplo

* Os dados foram padronizados para que as distâncias não fossem distorcidas.
* O modelo obteve um valor de **acurácia**, que mede o quanto ele classificou corretamente no teste.
* A fronteira de decisão mostra visualmente como o algoritmo divide o espaço entre as classes de câmbio.

---

**Conclusão**

O exemplo de KNN ilustra bem as propriedades fundamentais do método: a padronização (`StandardScaler`) foi necessária para que `km` e `price` contribuíssem de forma equilibrada nas distâncias. A métrica impressa (`accuracy`) fornece uma visão rápida do desempenho, mas não conta toda a história — especialmente quando há classes desbalanceadas ou custos diferentes para erros.

Pontos-chave para interpretar os resultados:

- **Acurácia**: indica a proporção de previsões corretas no conjunto de teste, mas deve ser complementada por outras métricas (matriz de confusão, precisão/recall, F1) quando as classes não são igualmente representadas.
- **Fronteira de decisão**: mostra como o espaço é dividido com `k=3`; regiões conectadas e suaves indicam boa separabilidade local, enquanto regiões muito irregulares podem sinalizar sensibilidade a ruído.
- **Importância da padronização**: sem escalar as features, variáveis com escala maior dominariam a distância euclidiana e prejudicariam o resultado.


