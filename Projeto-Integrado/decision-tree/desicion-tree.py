import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# ==========================
# PREPROCESSAMENTO
# ==========================
df = pd.read_csv('base_Projeto_integrado.csv')

def preprocess(df):
    df = df.copy()

    # Codificar variáveis categóricas
    encoder = LabelEncoder()
    df['combustivel'] = encoder.fit_transform(df['combustivel'])
    df['cambio'] = encoder.fit_transform(df['cambio'])
    df['modelo_base'] = encoder.fit_transform(df['modelo_base'])

    # Seleção de features
    features = ['km', 'combustivel', 'cambio', 'modelo_base']
    X = df[features]
    y = df['price']        # alvo = preço do carro

    return X, y



# Preprocessar
X, y = preprocess(df)

# Dividir os dados
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# TREINAR MODELO
# ==========================
regressor = tree.DecisionTreeRegressor(random_state=42)
regressor.fit(x_train, y_train)

# ==========================
# AVALIAR MODELO
# ==========================
y_pred = regressor.predict(x_test)
score = r2_score(y_test, y_pred)
print(f"R² do modelo: {score:.2f}")

# ==========================
# PLOTAR A ÁRVORE
# ==========================
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
