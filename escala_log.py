import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def df_to_pdf(df, path="escala_log_output.pdf"):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis('tight')
    ax.axis('off')
    tabla = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1.2, 1.2)
    
    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    
    print(f"DataFrame guardado en '{path}'")

def transformar_log(ruta_csv, nombre_columna, output_pdf_path="escala_log_output.pdf"):
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
        columna_datos = df[nombre_columna].astype(float)
        df[f'{nombre_columna}_log'] = np.log1p(columna_datos)
        
        df_to_pdf(df, path=output_pdf_path)
        
        return df
    except ValueError:
        print(f"Error: La columna '{nombre_columna}' no pudo ser convertida a valores numéricos.")
        return None
    except Exception as e:
        print(f"Ocurrió un error durante la transformación: {e}")
        return None

if __name__ == '__main__':
    data = {'col1': range(10), 'col2': np.random.rand(10) * 100}
    test_csv_path = 'test_data_log.csv'
    pd.DataFrame(data).to_csv(test_csv_path, index=False)
    
    print(f"Ejecutando ejemplo con el archivo '{test_csv_path}' y la columna 'col2'")
    transformar_log(test_csv_path, 'col2')
