# Modelo K-Nearest Neighbors (KNN)
## Objetivo

O modelo de K-Nearest Neighbors foi utilizado para prever o tipo de emprego (Job Title) de profissionais com base em suas características, como idade, anos de experiência, gênero e nível educacional. O objetivo é avaliar se o algoritmo consegue identificar padrões que diferenciem os cargos a partir desses atributos.

``` python exec="on" html="1"
--8<-- "docs/k-nearest-neighbor/knn.py"
```
## Resultados

O modelo obteve uma acurácia de aproximadamente X% (valor exibido na saída do código).

Foi gerado um gráfico de dispersão com base em idade normalizada e anos de experiência normalizados, colorindo cada ponto conforme o tipo de emprego.

## Conclusão

O KNN apresentou desempenho razoável para classificação de tipos de emprego a partir de atributos individuais.

Pontos fortes:
    Fácil implementação.
    Bom para problemas com distribuição não linear.
    Intuitivo e interpretável.

Limitações:
    Sensível à escala dos dados e à escolha de k.
    Pode ter dificuldade em datasets com muitas classes (como diversos títulos de emprego).
    Acurácia pode ser impactada por variáveis pouco informativas.

Como próximos passos, recomenda-se:
    1. Testar diferentes valores de k (ex: 5, 7, 9) e comparar resultados.
    2. Normalizar todas as variáveis (não só para plotagem), melhorando a performance do algoritmo.
    3. Comparar o desempenho do KNN com outros classificadores, como Árvore de Decisão ou Random Forest.