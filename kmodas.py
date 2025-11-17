import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages
import io
import sys

def text_to_pdf(text, pdf):
    """Agrega texto a una página en un PDF."""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 size
    plt.axis('off')
    plt.text(0.05, 0.95, text, va='top', ha='left', wrap=True, fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)

def run_kmodas(file_path, output_pdf_path="kmodas_output.pdf"):
    """
    Ejecuta el algoritmo K-Modas y guarda los resultados (gráficos y texto) en un archivo PDF.
    """
    dataset = pd.read_csv(file_path)
    X = dataset[['X2']].values

    # Redirigir stdout para capturar la salida de texto
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    with PdfPages(output_pdf_path) as pdf:
        # --- Gráfico del Método del Codo ---
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        fig1 = plt.figure()
        plt.plot(range(1, 11), wcss, marker='o')
        plt.title('Método del Codo')
        plt.xlabel('Número de clusters')
        plt.ylabel('WCSS')
        pdf.savefig(fig1)
        plt.close(fig1)

        # --- Clustering y Resultados ---
        kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
        dataset['Cluster_X2'] = kmeans.fit_predict(X)

        print("--- Resultados del Clustering K-Modas ---")
        print("\nCentroides:", kmeans.cluster_centers_.flatten())
        print("\nDataset con Clusters:")
        print(dataset)

        # --- Gráfico de Clusters ---
        fig2 = plt.figure()
        plt.scatter(X[dataset['Cluster_X2'] == 0], [0]*len(X[dataset['Cluster_X2'] == 0]), color='red', label='Cluster 1')
        plt.scatter(X[dataset['Cluster_X2'] == 1], [0]*len(X[dataset['Cluster_X2'] == 1]), color='blue', label='Cluster 2')
        plt.scatter(X[dataset['Cluster_X2'] == 2], [0]*len(X[dataset['Cluster_X2'] == 2]), color='green', label='Cluster 3')
        plt.scatter(kmeans.cluster_centers_, [0]*3, s=200, c='yellow', label='Centroides')
        plt.title('Clusters según X2')
        plt.xlabel('Valor de X2')
        plt.legend()
        pdf.savefig(fig2)
        plt.close(fig2)

        # --- Guardar texto capturado en el PDF ---
        sys.stdout = old_stdout  # Restaurar stdout
        output_text = captured_output.getvalue()
        text_to_pdf(output_text, pdf)
    
    print(f"Resultados de K-Modas guardados en '{output_pdf_path}'")


if __name__ == '__main__':
    # Reemplaza 'your_data.csv' con la ruta a tu archivo CSV
    run_kmodas('your_data.csv')
