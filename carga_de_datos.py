from datasets import load_dataset
import pandas as pd

# Cargar el dataset
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Dataset a un DataFrame
df = pd.DataFrame(data)

# Aqui se separa el DataFrame en dos, uno con personas que ya se murieron y otro con el complemento
df_dead = df[df['is_dead'] == 1]
df_alive = df[df['is_dead'] == 0]

# Promedio de las edades de cada dataset
promedio_edad_dead = df_dead['age'].mean()
promedio_edad_alive = df_alive['age'].mean()

print(f"Promedio de edad de las personas que perecieron: {promedio_edad_dead:.2f} años")
print(f"Promedio de edad de las personas que sobrevivieron: {promedio_edad_alive:.2f} años")