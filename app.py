from flask import Flask, request, jsonify, send_from_directory
from flasgger import Swagger
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.DEBUG)

try:
    import kmedias as kme
    import kmodas as kmo
    import arbol as tree
    import chimerge as cm
    import estandarizacion as est
    import normalizacion as norm
    import escala_log as log
except ImportError as e:
    logging.error(f"Error al importar módulos de algoritmos: {e}")

app = Flask("AnalyticaPro")
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SWAGGER'] = {
    'title': 'AnalyticaPro API',
    'uiversion': 3,
    "specs_route": "/algoritmos/"
}
swagger = Swagger(app)

@app.route('/')
def welcome():
    return 'Bienvenidx a AnalyticaPro'

@app.route('/download/<path:filename>')
def download_file(filename):
    app.logger.info(f"Solicitud de descarga para el archivo: {filename}")
    try:
        return send_from_directory(
            app.config['UPLOAD_FOLDER'],
            filename,
            as_attachment=True
        )
    except FileNotFoundError:
        app.logger.error(f"Archivo no encontrado en la ruta: {os.path.join(app.config['UPLOAD_FOLDER'], filename)}")
        return jsonify({"error": "Archivo no encontrado"}), 404

@app.route('/algoritmos', methods=['POST'])
def run_algorithm():
    """
    Run a specified data analysis algorithm via file upload.
    ---
    tags:
      - Algorithms
    consumes:
      - multipart/form-data
    parameters:
      - name: data_file
        in: formData
        type: file
        required: true
        description: The CSV data file to process.
      - name: algoritmo
        in: formData
        type: string
        required: true
        description: The algorithm to execute.
        enum: ['ESTANDARIZACION', 'NORMALIZACION', 'ESCALA_LOG', 'CHIMERGE', 'KMODAS', 'KMEDIAS', 'ARBOL']
      - name: output_filename
        in: formData
        type: string
        required: false
        description: Optional desired name for the output PDF file.
      - name: nombre_columna
        in: formData
        type: string
        description: Name of the column to process (for ESTANDARIZACION, NORMALIZACION, ESCALA_LOG).
      - name: objetivo
        in: formData
        type: string
        description: Name of the target column (for ARBOL).
      - name: inicio
        in: formData
        type: string
        description: Name of the starting column for analysis range (for ARBOL).
    responses:
      200:
        description: Algorithm executed successfully.
      400:
        description: Bad request.
    """
    app.logger.debug(f"Formulario (request.form): {request.form.to_dict()}")
    app.logger.debug(f"Archivos (request.files): {request.files.to_dict()}")

    if 'data_file' not in request.files:
        return jsonify({"error": "Falta el archivo 'data_file' en la solicitud."}), 400

    file = request.files['data_file']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo."}), 400

    algoritmo = request.form.get('algoritmo')
    if not algoritmo:
        return jsonify({"error": "Falta el parámetro 'algoritmo'."}), 400

    input_filename = secure_filename(file.filename)
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    file.save(data_path)
    app.logger.info(f"Archivo guardado en: {data_path}")

    user_defined_output = request.form.get('output_filename')
    if user_defined_output:
        # Sanitize the user-provided filename
        output_filename = secure_filename(user_defined_output)
        # Ensure it has a .pdf extension
        if not output_filename.lower().endswith('.pdf'):
            output_filename += '.pdf'
    else:
        # Fallback to the original naming convention
        output_filename = f"{os.path.splitext(input_filename)[0]}_{algoritmo.lower()}_output.pdf"

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    result = None
    error_message = None

    try:
        if algoritmo in ['ESTANDARIZACION', 'NORMALIZACION', 'ESCALA_LOG']:
            nombre_columna = request.form.get('nombre_columna')
            if not nombre_columna:
                return jsonify({"error": f"Falta el parámetro 'nombre_columna' para {algoritmo}"}), 400

            if algoritmo == 'ESTANDARIZACION':
                result = est.estandarizar_datos(data_path, nombre_columna, output_pdf_path=output_path)
                if result is None: error_message = f"La columna '{nombre_columna}' no se encontró en el archivo o no es numérica."
            elif algoritmo == 'NORMALIZACION':
                result = norm.normalizar_datos(data_path, nombre_columna, output_pdf_path=output_path)
                if result is None: error_message = f"La columna '{nombre_columna}' no se encontró en el archivo o no es numérica."
            elif algoritmo == 'ESCALA_LOG':
                result = log.transformar_log(data_path, nombre_columna, output_pdf_path=output_path)
                if result is None: error_message = f"La columna '{nombre_columna}' no se encontró en el archivo o no es numérica."

        elif algoritmo == 'CHIMERGE':
            result = cm.run_chimerge(data_path, output_pdf_path=output_path)
            if result is None: error_message = "Error durante la ejecución de Chi-Merge. Verifique el formato del archivo y las columnas."

        elif algoritmo == 'KMODAS':
            result = kmo.run_kmodas(data_path, output_pdf_path=output_path)
            if result is None: error_message = "Error en K-Modas. Asegúrese de que la columna 'X2' exista y sea numérica."

        elif algoritmo == 'KMEDIAS':
            result = kme.run_kmedias(data_path, output_pdf_path=output_path)
            if result is None: error_message = "Error en K-Medias. Verifique las columnas 'longitude', 'latitude', y 'median_house_value'."

        elif algoritmo == 'ARBOL':
            objetivo = request.form.get('objetivo')
            inicio = request.form.get('inicio')
            if not all([objetivo, inicio]):
                return jsonify({"error": "Faltan los parámetros 'objetivo' o 'inicio' para ARBOL"}), 400

            encabezado, datos = tree.cargar_csv(data_path)
            if not encabezado or not datos:
                return jsonify({"error": "No se pudieron cargar los datos del CSV"}), 500

            if objetivo not in encabezado or inicio not in encabezado:
                return jsonify({"error": f"Las columnas '{objetivo}' o '{inicio}' no se encuentran en el archivo"}), 400

            idx_final_int = encabezado.index(objetivo)
            idx_inicio_int = encabezado.index(inicio)
            indices_vars = list(range(idx_inicio_int, idx_final_int))

            arbol_resultado = tree.construir_arbol(datos, encabezado, indices_vars, idx_final_int)
            
            # For ARBOL, we might have multiple outputs, so we handle it slightly differently
            output_filename_visual = output_path.replace('.pdf', '_visual.pdf')
            tree.dibujar_arbol_pdf(arbol_resultado, output_filename_visual)
            
            result = True 
            output_filename = os.path.basename(output_filename_visual)

        else:
            return jsonify({"error": f"Algoritmo desconocido: {algoritmo}"}), 400

        if result is None:
            app.logger.error(f"La ejecución del algoritmo falló: {error_message}")
            return jsonify({"error": error_message}), 400

        app.logger.info(f"Ejecución exitosa. Archivo de salida: {output_filename}")
        return jsonify({
            "message": f"{algoritmo} ejecutado con éxito.",
            "output_filename": output_filename
        })

    except Exception as e:
        app.logger.exception("Ocurrió una excepción durante la ejecución del algoritmo")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
