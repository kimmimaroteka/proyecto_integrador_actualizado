import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

def procesar_datos():
    # Cargar el CSV limpio
    df = pd.read_csv('datos_limpios.csv')

    # Eliminar las columnas 'is_dead' y 'categoria_edad'
    X = df.drop(columns=['is_dead', 'categoria_edad']).values

    # Exportar un array unidimensional de la columna 'is_dead'
    y = df['is_dead'].values

    # Ejecutar t-SNE
    X_embedded = TSNE(
        n_components=3,
        learning_rate='auto',
        init='random',
        perplexity=3
    ).fit_transform(X)

    # Realizar un gráfico de dispersión 3D con Plotly
    fig = px.scatter_3d(
        x=X_embedded[:, 0], 
        y=X_embedded[:, 1], 
        z=X_embedded[:, 2],
        color=y,
        labels={'color': 'is_dead'},
        title="Visualización 3D con t-SNE"
    )

    fig.show()

if __name__ == "__main__":
    procesar_datos()