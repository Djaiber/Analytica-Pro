import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def df_to_pdf(df, path="estandarizacion_output.pdf"):
    """Guarda un DataFrame de pandas en un archivo PDF."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69)) # Tamaño A4
    ax.axis('tight')
    ax.axis('off')
    tabla = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1.2, 1.2)
    
    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    
    print(f"DataFrame guardado en '{path}'")

def estandarizar_datos(ruta_csv, nombre_columna, output_pdf_path="estandarizacion_output.pdf"):
    """
    Estandariza una columna de un CSV y guarda el DataFrame resultante en un PDF.
    """
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_csv}' no fue encontrado.")
        return None
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return None

    if nombre_columna not in df.columns:
        print(f"Error: La columna '{nombre_columna}' no se encuentra en el archivo.")
        return None

    try:
        scaler = StandardScaler()
        columna_datos = df[[nombre_columna]].values.astype(float)
        df[f'{nombre_columna}_z'] = scaler.fit_transform(columna_datos)
        
        # Guardar el DataFrame resultante en un PDF
        df_to_pdf(df, path=output_pdf_path)
        
        return df
    except ValueError:
        print(f"Error: La columna '{nombre_columna}' no pudo ser convertida a valores numéricos.")
        return None
    except Exception as e:
        print(f"Ocurrió un error durante la estandarización: {e}")
        return None

if __name__ == '__main__':
    # Ejemplo de uso:
    # estandarizar_datos('your_data.csv', 'column_name')

    # Para ejecutar un ejemplo, creemos un CSV de prueba
    data = {'id': range(10), 'puntuacion': [88, 92, 80, 89, 95, 85, 79, 91, 82, 90]}
    test_csv_path = 'test_data_std.csv'
    pd.DataFrame(data).to_csv(test_csv_path, index=False)
    
    print(f"Ejecutando ejemplo con el archivo '{test_csv_path}' y la columna 'puntuacion'")
    estandarizar_datos(test_csv_path, 'puntuacion')
