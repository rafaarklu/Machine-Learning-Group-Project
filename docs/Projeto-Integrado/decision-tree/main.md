# Árvore de Decisão (Decision Tree)

## O que é uma Árvore de Decisão?

A Árvore de Decisão é um modelo supervisionado que busca aprender padrões presentes nos dados através de divisões sucessivas, criando uma estrutura hierárquica semelhante a um fluxograma. Cada divisão ocorre com base na feature mais relevante para separar os valores do alvo.

Quando usada para **regressão**, como neste exemplo, o objetivo é prever um valor contínuo — neste caso, o preço do carro.

### Vantagens:

* Fácil interpretação.
* Não exige padronização dos dados.
* Captura relações não lineares.

### Desvantagens:

* Pode sofrer overfitting.
* Pequenas variações no dataset afetam a estrutura.

---

## Exemplo de Execução (Regressão)



```python exec="on" html="1"
--8<-- "docs\Projeto-Integrado\decision-tree\desicion-tree.py"
```


O código a seguir carrega dados, pré-processa variáveis categóricas, treina uma árvore e gera o valor **R²**, métrica que mostra o quanto o modelo explica da variabilidade do alvo.

### Código Base Utilizado

```python
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

df = pd.read_csv('base_Projeto_integrado.csv')

# Pré-processamento
encoder = LabelEncoder()
df['combustivel'] = encoder.fit_transform(df['combustivel'])
df['cambio'] = encoder.fit_transform(df['cambio'])
df['modelo_base'] = encoder.fit_transform(df['modelo_base'])

X = df[['km','combustivel','cambio','modelo_base']]
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treino
regressor = tree.DecisionTreeRegressor(random_state=42)
regressor.fit(x_train, y_train)

# Avaliação
pred = regressor.predict(x_test)
score = r2_score(y_test, pred)
print(f"R² do modelo: {score:.2f}")

# Plot da árvore
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
```

### Interpretação do Exemplo

* A árvore aprendeu a relacionar **km**, tipo de combustível, câmbio e modelo do carro ao preço.
* O valor de **R²** indica o quão bem a árvore se ajustou aos dados.
* O gráfico gerado mostra visualmente a estrutura da árvore e como ela divide as features para chegar a estimativas.

---

**Conclusão**

A aplicação da Árvore de Decisão ao conjunto de dados permite observar relações claras entre as features selecionadas (`km`, `combustivel`, `cambio`, `modelo_base`) e o preço dos veículos. A métrica impressa pelo script (`R²`) é o indicador primário para avaliar o ajuste: valores próximos de 1 significam que o modelo explica bem a variabilidade do preço, enquanto valores perto de 0 indicam que grande parte da variabilidade não foi capturada.

Por ser um modelo de alta interpretabilidade, a árvore é uma boa opção inicial para entender quais variáveis têm maior impacto nas previsões. Entretanto, árvores de decisão individuais são sensíveis a overfitting e a ruído nos dados. 

