import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def cargar_datos(archivo):
    df = pd.read_csv(archivo)
    return df

def preprocesar_datos(df):
    df = df.dropna(subset=['ExperienceInCurrentDomain', 'JoiningYear'])
    df['PaymentTier'].fillna(df['PaymentTier'].mode()[0], inplace=True)
    for col in ['Age', 'ExperienceInCurrentDomain', 'JoiningYear']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    df['CategoriaEdad'] = pd.cut(df['Age'], bins=[0, 12, 19, 39, 59, 100], labels=['Niño', 'Adolescente', 'Joven Adulto', 'Adulto', 'Adulto Mayor'])
    return df

def predecir_edades():
    # Cargar los datos
    df = cargar_datos('empleados.csv')
    df = preprocesar_datos(df)

    # Eliminación de las columnas solicitadas
    X = df.drop(columns=['LeaveOrNot', 'Age', 'CategoriaEdad'])
    y = df['Age']

    # De categóricas a variables dummy
    X = pd.get_dummies(X, drop_first=True)

    # Datos en conjuntos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo de Regresión Lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Prediccion de las Edads
    y_pred = modelo.predict(X_test)

    # Error cuadrático
    mse = mean_squared_error(y_test, y_pred)
    print(f'Error Cuadrático Medio: {mse:.2f}')

    # Mostrar las primeras filas de las edades reales y las predichas
    comparacion = pd.DataFrame({'Edad Real': y_test, 'Edad Predicha': y_pred})
    print(comparacion.head())

if __name__ == "__main__":
    predecir_edades()
