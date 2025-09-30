# Modelo K-Means Clustering
## Objetivo
O modelo de K-Means foi utilizado para agrupar profissionais em clusters com base em anos de experiência e salário. O objetivo é identificar padrões e segmentar perfis semelhantes, permitindo entender como diferentes níveis de experiência se relacionam com a remuneração.

``` python exec="on" html="1"
--8<-- "docs/k-means/kmeans.py"
```

# Metodologia

### 1 Seleção de variáveis

Foram escolhidas as colunas Years of Experience e Salary como base para o agrupamento.

### 2 Tratamento de dados ausentes

As linhas com valores nulos nessas colunas foram removidas para evitar inconsistências no clustering.

### 3 Configuração do algoritmo

Número de clusters: 3.

Inicialização: k-means++ (para melhor escolha dos centróides iniciais).

Máx. iterações: 100.

random_state = 42 para reprodutibilidade.

### 4 Clusters e visualização

Cada cluster foi representado com uma cor distinta.

Os centróides foram destacados em vermelho no gráfico.

Para melhor interpretação, foi exibido também o nível de educação mais comum dentro de cada cluster (quando disponível).


## Visualização

Os pontos foram agrupados em 3 clusters distintos.

Os centróides representam a média de cada cluster em termos de anos de experiência e salário.

O gráfico facilita a identificação de faixas salariais associadas a diferentes níveis de experiência.

## Conclusão

O uso de K-Means permitiu segmentar os profissionais em grupos distintos:

Cluster com baixa experiência e salários iniciais.

Cluster intermediário com experiência média e salários medianos.

Cluster avançado com maior experiência e salários elevados.

Esse tipo de agrupamento é útil para análises de mercado de trabalho, ajudando a identificar perfis típicos de profissionais e possíveis discrepâncias salariais.