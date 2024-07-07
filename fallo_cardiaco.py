from datasets import load_dataset
import numpy as np

# Cargar el dataset
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Extraer la lista de edades
edades = data["age"]

# Convertir la lista de edades a un arreglo de NumPy
edades_np = np.array(edades)

# Calcular el promedio de edad
promedio_edad = np.mean(edades_np)

# Mostrar el promedio de edad
print(f"El promedio de edad de las personas participantes en el estudio es: {promedio_edad:.2f} años")