import pandas as pd
import matplotlib.pyplot as plt

def graficar_pasteles():
    df = pd.read_csv('datos_limpios.csv')

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    categorias = ['has_anaemia', 'has_diabetes', 'is_smoker', 'is_dead']
    titulos = ['Anémicos', 'Diabéticos', 'Fumadores', 'Muertos']

    colores = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

    for ax, categoria, titulo in zip(axs, categorias, titulos):
        conteo = df[categoria].value_counts()
        ax.pie(conteo, labels=['No', 'Sí'], autopct='%1.1f%%', colors=colores[:2])
        ax.set_title(titulo)

    plt.tight_layout()
    plt.savefig('graficas_pastel.png')
    plt.show()

if __name__ == "__main__":
    graficar_pasteles()