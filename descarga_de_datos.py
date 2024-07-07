import requests

def descargar_datos(url, nombre_archivo):
    try:
        response = requests.get(url)
        
        response.raise_for_status()
        
        # .csv
        with open(nombre_archivo, 'w') as archivo:
            archivo.write(response.text)
        
        print(f"Datos guardados exitosamente en {nombre_archivo}")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar los datos: {e}")

# URL
url = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
# Nombre del archivo donde se guardarán los datos
nombre_archivo = "heart_failure_clinical_records_dataset.csv"

# Llamar a la función para descargar y guardar los datos
descargar_datos(url, nombre_archivo)