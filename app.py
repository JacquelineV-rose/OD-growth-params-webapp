import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from src.DataTransformer import TecanDataTransformer
from src.Wellplate import Wellplate

import pandas as pd
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['csvfile']
        if file.filename == '':
            return render_template('index.html', error='No file selected.')

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            data = TecanDataTransformer.load_data(filepath)
            transformed = TecanDataTransformer.transform_data(data)
            well_data = TecanDataTransformer.get_transformed_data(transformed)

            plate = Wellplate((16, 24), well_data)

            unique_id = str(uuid.uuid4())
            image_path = f'{RESULTS_FOLDER}/plot_{unique_id}.png'
            csv_path = f'{RESULTS_FOLDER}/results_{unique_id}.tsv'

            plate.plot_raw_data(save_path=image_path)
            plate.output_csv(csv_path)

            return render_template('index.html', plot_image=image_path, results_file=csv_path)
        except Exception as e:
            return render_template('index.html', error=f'Processing error: {str(e)}')

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)