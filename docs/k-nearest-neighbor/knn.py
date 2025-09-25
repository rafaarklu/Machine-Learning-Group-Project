import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.calibration import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

plt.figure(figsize=(10, 6))

# Limpar e preparar os dados
def preprocess(df):
    # Fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Years of Experience'].fillna(df['Years of Experience'].median(), inplace=True)
    df['Salary'].fillna(df['Salary'].median(), inplace=True)

    # Convert categorical variables
    label_encoder_gender = LabelEncoder()
    label_encoder_education = LabelEncoder()
    label_encoder_job = LabelEncoder()
    
    df['Gender_encoded'] = label_encoder_gender.fit_transform(df['Gender'])
    df['Education_encoded'] = label_encoder_education.fit_transform(df['Education Level'])
    df['Job_encoded'] = label_encoder_job.fit_transform(df['Job Title'])
    
    return df

# Load salary dataset
df = pd.read_csv('https://raw.githubusercontent.com/rafaarklu/Machine-Learning-Group-Project/refs/heads/main/Salary_Data.csv')
df = preprocess(df)
X = df[['Age', 'Years of Experience', 'Gender_encoded', 'Education_encoded']]
y = df['Job_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")


df_plot = pd.DataFrame()
df_plot['Age'] = (X['Age']-X['Age'].min())/(X['Age'].max()-X['Age'].min())
df_plot['Experience'] = (X['Years of Experience']-X['Years of Experience'].min())/(X['Years of Experience'].max()-X['Years of Experience'].min())
df_plot['Job_Type'] = y
sns.scatterplot(data=df_plot, x='Age', y='Experience', hue='Job_Type')


#Ver se é necessario dps

# Visualize decision boundary
#h = 0.02  # Step size in mesh
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
#Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)
#
#plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
#sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, style=y, palette="deep", s=100)
#plt.xlabel("Feature 1")
#plt.ylabel("Feature 2")
#plt.title("KNN Decision Boundary (k=3)")
#
## Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())