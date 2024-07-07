import requests
import pandas as pd
import argparse

def descargar_datos(url, nombre_archivo):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(nombre_archivo, 'w') as archivo:
            archivo.write(response.text)
        print(f"Datos guardados exitosamente en {nombre_archivo}")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar los datos: {e}")
        raise

def limpiar_y_preparar_datos(df):
    # Verificar y eliminar valores faltantes
    if df.isnull().values.any():
        df = df.dropna()
        print("Se eliminaron valores faltantes.")

    # Verificar y eliminar filas repetidas
    if df.duplicated().any():
        df = df.drop_duplicates()
        print("Se eliminaron filas repetidas.")

    # Verificar y eliminar valores atípicos
    Q1 = df['age'].quantile(0.25)
    Q3 = df['age'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['age'] < (Q1 - 1.5 * IQR)) | (df['age'] > (Q3 + 1.5 * IQR)))]
    print("Se eliminaron valores atípicos.")

    # Crear una columna que categorice por edades
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

def main(url):
    # Nombre del archivo donde se guardarán los datos descargados
    nombre_archivo = "heart_failure_clinical_records_dataset.csv"

    # Descargar y guardar los datos
    descargar_datos(url, nombre_archivo)

    # Leer los datos descargados en un DataFrame
    df = pd.read_csv(nombre_archivo)

    # Limpiar y preparar los datos
    df_limpio = limpiar_y_preparar_datos(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Descargar y procesar datos de fallo cardíaco.')
    parser.add_argument('url', type=str, help='La URL del archivo CSV con los datos de fallo cardíaco.')
    args = parser.parse_args()

    main(args.url)