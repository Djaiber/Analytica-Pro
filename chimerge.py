import pandas as pd
import sys
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def text_to_pdf(text, pdf):
    """Agrega texto a una página en un PDF."""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 size
    plt.axis('off')
    plt.text(0.05, 0.95, text, va='top', ha='left', wrap=True, fontsize=8, family='monospace')
    pdf.savefig(fig)
    plt.close(fig)

def run_chimerge(file_path, output_pdf_path="chimerge_output.pdf", num_intervals_deseados1=3, num_intervals_deseados2=3):
    """
    Ejecuta la discretización Chi-Merge y guarda la salida de texto en un archivo PDF.
    """
    # Redirigir stdout para capturar la salida
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    print('--- Iniciando Discretización Chi-Merge ---')

    try:
        df = pd.read_csv(file_path)
        df['X'] = pd.to_numeric(df['X'])
        df['Y'] = pd.to_numeric(df['Y'])
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no fue encontrado.", file=sys.stderr)
        sys.stdout = old_stdout # Restaurar stdout en caso de error
        print(captured_output.getvalue(), file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Error de datos: {e}", file=sys.stderr)
        sys.stdout = old_stdout # Restaurar stdout
        print(captured_output.getvalue(), file=sys.stderr)
        sys.exit(1)

    classes = df['CLASE'].unique().tolist()
    if len(classes) != 2:
        print(f"Error: Se requieren 2 clases, pero se encontraron {len(classes)}: {classes}", file=sys.stderr)
        sys.stdout = old_stdout # Restaurar stdout
        print(captured_output.getvalue(), file=sys.stderr)
        sys.exit(1)

    # --- Procesar y capturar salida ---
    discretize_column(df, 'X', 'CLASE', num_intervals_deseados1, classes)
    print("\n" + "="*50 + "\n") # Separador
    discretize_column(df, 'Y', 'CLASE', num_intervals_deseados2, classes)

    # --- Restaurar stdout y obtener el texto ---
    sys.stdout = old_stdout
    output_text = captured_output.getvalue()

    # --- Guardar el texto en un PDF ---
    with PdfPages(output_pdf_path) as pdf:
        text_to_pdf(output_text, pdf)
    
    print(f"Resultados de Chi-Merge guardados en '{output_pdf_path}'")
    # Opcional: imprimir también en la consola
    # print(output_text)


def discretize_column(df, feature_col, class_col, num_intervals_deseados, classes):
    """Imprime los pasos de la discretización Chi-Merge para una columna."""
    print(f"--- Discretizando columna '{feature_col}' ---")
    
    data = df[[feature_col, class_col]].sort_values(by=feature_col)
    unique_vals = data[feature_col].unique()

    intervals_data = [list(data[data[feature_col] == val].itertuples(index=False, name=None)) for val in unique_vals]
    intervals_repr = [f"[{val},{val}]" for val in unique_vals]

    print(f"Intervalos iniciales ({len(intervals_repr)}): {intervals_repr}\n")

    while len(intervals_repr) > num_intervals_deseados:
        chi_squares = [calculate_chi_square(intervals_data[i], intervals_data[i+1], classes) for i in range(len(intervals_data) - 1)]

        if not chi_squares:
            break

        min_chi = min(chi_squares)
        min_index = chi_squares.index(min_chi)

        # Imprimir el paso de fusión
        print(f"Fusionando {intervals_repr[min_index]} y {intervals_repr[min_index+1]} (Chi-cuadrado = {min_chi:.4f})")

        # Fusionar
        intervals_data[min_index].extend(intervals_data[min_index + 1])
        del intervals_data[min_index + 1]

        start_val = intervals_repr[min_index].split(',')[0][1:]
        end_val = intervals_repr[min_index + 1].split(',')[1][:-1]
        intervals_repr[min_index] = f"[{start_val},{end_val}]"
        del intervals_repr[min_index + 1]

    print(f'\nIntervalos finales para {feature_col}:')
    print(intervals_repr)


def calculate_chi_square(interval1, interval2, classes):
    """Calcula el valor de chi-cuadrado entre dos intervalos."""
    class1, class2 = classes[0], classes[1]
    A1 = sum(1 for _, c in interval1 if c == class1)
    B1 = sum(1 for _, c in interval1 if c == class2)
    A2 = sum(1 for _, c in interval2 if c == class1)
    B2 = sum(1 for _, c in interval2 if c == class2)

    sum_row1, sum_row2 = A1 + B1, A2 + B2
    sum_colA, sum_colB = A1 + A2, B1 + B2
    total = sum_row1 + sum_row2

    if total == 0: return 0.0

    chi_sq = 0.0
    for obs, s_row, s_col in [(A1, sum_row1, sum_colA), (B1, sum_row1, sum_colB), 
                               (A2, sum_row2, sum_colA), (B2, sum_row2, sum_colB)]:
        if s_row > 0 and s_col > 0:
            expected = (s_row * s_col) / total
            if expected > 0:
                chi_sq += ((obs - expected) ** 2) / expected
    return chi_sq


if __name__ == '__main__':
    run_chimerge('your_data.csv')
