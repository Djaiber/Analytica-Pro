import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.backends.backend_pdf import PdfPages

def run_kmedias(file_path, output_pdf_path="kmedias_output.pdf"):
    """
    Ejecuta un análisis de K-Medias y guarda todos los gráficos en un archivo PDF.
    """
    try:
        home_data = pd.read_csv(file_path, usecols=['longitude', 'latitude', 'median_house_value'])
    except FileNotFoundError:
        print(f"Error: El archivo '{file_path}' no fue encontrado.")
        return
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return

    with PdfPages(output_pdf_path) as pdf:
        # --- Gráfico 1: Visualización inicial de datos ---
        fig = plt.figure()
        sns.scatterplot(data=home_data, x='longitude', y='latitude', hue='median_house_value')
        plt.title('Distribución Geográfica vs. Valor Mediano de la Vivienda')
        pdf.savefig(fig)
        plt.close(fig)

        # --- Preparación de datos ---
        X_train, _, y_train, _ = train_test_split(
            home_data[['latitude', 'longitude']],
            home_data[['median_house_value']],
            test_size=0.33,
            random_state=0
        )
        X_train_norm = preprocessing.normalize(X_train)

        # --- Gráfico 2: Clustering inicial (k=3) ---
        kmeans_3 = KMeans(n_clusters=3, random_state=0, n_init='auto')
        kmeans_3.fit(X_train_norm)
        fig = plt.figure()
        sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=kmeans_3.labels_)
        plt.title('Clusters de Viviendas (k=3)')
        pdf.savefig(fig)
        plt.close(fig)

        # --- Gráfico 3: Boxplot de valor por cluster (k=3) ---
        fig = plt.figure()
        sns.boxplot(x=kmeans_3.labels_, y=y_train['median_house_value'])
        plt.title('Valor Mediano de Vivienda por Cluster (k=3)')
        pdf.savefig(fig)
        plt.close(fig)

        # --- Búsqueda del K óptimo ---
        K = range(2, 8)
        fits = []
        scores = []
        for k in K:
            model = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(X_train_norm)
            fits.append(model)
            scores.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))

        # --- Gráfico 4: Puntuación de Silueta vs. K ---
        fig = plt.figure()
        sns.lineplot(x=K, y=scores)
        plt.title('Puntuación de Silueta para Diferentes K')
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Puntuación de Silueta')
        pdf.savefig(fig)
        plt.close(fig)
        
        # Encontrar el mejor k según la puntuación de silueta
        best_k_index = scores.index(max(scores))
        best_k = K[best_k_index]
        best_fit = fits[best_k_index]

        # --- Gráfico 5: Mejor clustering según silueta ---
        fig = plt.figure()
        sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=best_fit.labels_)
        plt.title(f'Mejor Clustering Encontrado (k={best_k})')
        pdf.savefig(fig)
        plt.close(fig)

        # --- Gráfico 6: Boxplot del mejor clustering ---
        fig = plt.figure()
        sns.boxplot(x=best_fit.labels_, y=y_train['median_house_value'])
        plt.title(f'Valor Mediano de Vivienda por Cluster (k={best_k})')
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Análisis de K-Medias completado. Gráficos guardados en '{output_pdf_path}'")

if __name__ == '__main__':
    # Reemplaza 'housing.csv' con la ruta a tu archivo de datos.
    run_kmedias('housing.csv')
