from datasets import load_dataset
import pandas as pd

# dataset
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Dataset a DataFrame
df = pd.DataFrame(data)

# Verificar los Tipos de Datos
print("Tipos de datos en el DataFrame:")
print(df.dtypes)

# Aqui se calcula la cantidad de hombres que fuman y las mujeres que fuman mucho

fumar_por_genero = df.groupby(['is_male', 'is_smoker']).size().unstack(fill_value=0)

print("\nCantidad de hombres fumadores vs mujeres fumadoras:")
print(fumar_por_genero)