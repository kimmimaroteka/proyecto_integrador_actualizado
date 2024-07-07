import pandas as pd
import matplotlib.pyplot as plt

def graficar_datos():
    df = pd.read_csv('datos_limpios.csv')

    plt.figure(figsize=(10, 6))
    plt.hist(df['age'], bins=20, edgecolor='black')
    plt.title('Distribución de Edades')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.savefig('histograma_edad.png')
    plt.show()

    categorias = ['has_anaemia', 'has_diabetes', 'is_smoker', 'is_dead']
    titulos = ['Cantidad de Anémicos', 'Cantidad de Diabéticos', 'Cantidad de Fumadores', 'Cantidad de Muertos']
    colores = ['blue', 'orange']

    for categoria, titulo in zip(categorias, titulos):
        hombres = df[df['is_male'] == 1][categoria].value_counts().sort_index()
        mujeres = df[df['is_male'] == 0][categoria].value_counts().sort_index()

        hombres = hombres.reindex([0, 1], fill_value=0)
        mujeres = mujeres.reindex([0, 1], fill_value=0)

        indices = [0, 1]
        bar_width = 0.4

        plt.figure(figsize=(10, 6))
        plt.bar(indices, hombres, width=bar_width, color=colores[0], align='edge', label='Hombres')
        plt.bar([i - bar_width for i in indices], mujeres, width=-bar_width, color=colores[1], align='edge', label='Mujeres')

        plt.title(titulo)
        plt.xlabel(categoria)
        plt.ylabel('Frecuencia')
        plt.xticks(indices, ['No', 'Sí'])
        plt.legend()
        plt.savefig(f'histograma_{categoria}.png')
        plt.show()

if __name__ == "__main__":
    graficar_datos()