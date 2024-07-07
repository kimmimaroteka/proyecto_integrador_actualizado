import requests
import pandas as pd

def descargar_datos(url, nombre_archivo):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(nombre_archivo, 'w') as archivo:
            archivo.write(response.text)
        print(f"Datos guardados exitosamente en {nombre_archivo}")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar los datos: {e}")

def limpiar_y_preparar_datos(df):
    # Verifica y elimina los valores faltantes
    if df.isnull().values.any():
        df = df.dropna()
        print("Se eliminaron valores faltantes.")

    # Verifica y elimina las filas repetidas
    if df.duplicated().any():
        df = df.drop_duplicates()
        print("Se eliminaron filas repetidas.")

    # Verifica y elimina los valores atípicos
    Q1 = df['age'].quantile(0.25)
    Q3 = df['age'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['age'] < (Q1 - 1.5 * IQR)) | (df['age'] > (Q3 + 1.5 * IQR)))]
    print("Se eliminaron valores atípicos.")

    # Crea una columna que categorice por edades
    def categorizar_edad(edad):
        if edad <= 12:
            return 'Niño'
        elif edad <= 19:
            return 'Adolescente'
        elif edad <= 39:
            return 'Joven adulto'
        elif edad <= 59:
            return 'Adulto'
        else:
            return 'Adulto mayor'

    df['categoria_edad'] = df['age'].apply(categorizar_edad)

    # Guardar el resultado como csv
    nombre_archivo_limpio = 'datos_limpios.csv'
    df.to_csv(nombre_archivo_limpio, index=False)
    print(f"Datos limpios guardados en {nombre_archivo_limpio}")

    return df

# URL de los datos
url = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
# Nombre del archivo donde se van a guardar esos datos nuevos
nombre_archivo = "heart_failure_clinical_records_dataset.csv"

descargar_datos(url, nombre_archivo)

df = pd.read_csv(nombre_archivo)

df_limpio = limpiar_y_preparar_datos(df)