# Modelo Arvore de Decisão 
## Objetivo
O modelo de Árvore de Decisão Regressora foi implementado para prever salários com base em variáveis relacionadas ao perfil do profissional, como idade, gênero, nível educacional e anos de experiência. O objetivo é entender como essas variáveis impactam no salário e identificar padrões dentro dos dados.

```python exec="on" html="1"
--8<-- "docs/Decision-Tree/decision-tree.py"
```

## Avaliação do Modelo

Após o treinamento, o modelo foi avaliado utilizando Mean Squared Error (MSE) para medir a diferença entre os valores reais e previstos. Além disso, foi analisada a importância das features, que mostra quais variáveis mais influenciam no salário.

Exemplo esperado (a ordem pode variar de acordo com os dados):

!!! tip "Feature Importances"
    Years of Experience → maior impacto.<br>

    Education Level → influência intermediária.<br>

    Age → impacto menor.<br>

    Gender → pouca influência.
   
    


```python exec="on" html="1"
--8<-- "docs/Decision-Tree/Arvore.py"
```

## Visualização

### A árvore de decisão foi plotada com: 
Nós arredondados e coloridos (filled=True, rounded=True).
Nomes das variáveis exibidos.
Profundidade limitada para melhor interpretação.
Essa visualização facilita a compreensão dos caminhos de decisão do modelo, evidenciando quais fatores mais contribuem para a previsão dos salários.

## Conclusão 

O modelo de Árvore de Decisão Regressora demonstrou ser uma ferramenta útil para prever salários a partir de atributos individuais.
Ele fornece interpretação clara através da visualização da árvore.
Mostra a importância relativa das variáveis, auxiliando na análise de quais fatores mais influenciam o salário. Apesar de ser interpretável, deve-se ter cuidado com overfitting, já que árvores muito profundas podem se ajustar em excesso aos dados de treino.
Esse modelo pode ser expandido ou comparado com outros algoritmos (como Random Forest ou Regressão Linear) para verificar melhorias em precisão e generalização.