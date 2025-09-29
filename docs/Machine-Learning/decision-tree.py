import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Preprocess the data for REGRESSION
def preprocess_salary_data(df):
    """Prepara os dados para o modelo de Regressão, tratando NAs e codificando variáveis."""

    # Features selecionadas (excluindo 'Job Title' devido à alta cardinalidade)
    features = ['Age', 'Gender', 'Education Level', 'Years of Experience', 'Salary']
    df_clean = df[features].copy()

    # 1. Tratar Valores Ausentes (NA/NaN)
    # Numéricas -> Mediana
    for col in ['Age', 'Years of Experience', 'Salary']:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

    # Categóricas -> Moda
    for col in ['Gender', 'Education Level']:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    # 2. Converter variáveis categóricas usando One-Hot Encoding
    df_encoded = pd.get_dummies(df_clean, columns=['Gender', 'Education Level'], drop_first=True)

    return df_encoded

# Configura o tamanho do plot
plt.figure(figsize=(18, 12))

# Carregar o conjunto de dados
df = pd.read_csv('Salary_Data.csv')

# Preprocessar o conjunto de dados
df_processed = preprocess_salary_data(df)
x = df_processed.drop('Salary', axis=1) # Features
y = df_processed['Salary'] # Variável Alvo

# Dividir os dados em conjuntos de treinamento e teste (80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de árvore de decisão (REGRESSOR)
# Limitando a profundidade para melhor visualização e evitar overfitting extremo
regressor = tree.DecisionTreeRegressor(random_state=42, max_depth=5)
regressor.fit(x_train, y_train)

# Avaliar o modelo
y_pred = regressor.predict(x_test)


# Imprimir a importância das features
feature_importance = pd.DataFrame({
    'Feature': x.columns.tolist(),
    'Importance': regressor.feature_importances_
})
print("\n<br>Feature Importances:")
print(feature_importance.sort_values(by='Importance', ascending=False).to_html(index=False))

# Visualizar a árvore
tree.plot_tree(regressor,
               feature_names=x.columns.tolist(),
               filled=True,
               rounded=True,
               fontsize=8)

# Para imprimir a partir do StringIO (como solicitado)
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())