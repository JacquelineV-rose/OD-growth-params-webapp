import os
import uuid
from flask import Flask, request, render_template, send_from_directory
from src.DataTransformer import TecanDataTransformer
from src.Wellplate import Wellplate

import pandas as pd

app = Flask(__name__)

# Define folders for uploads and results
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

        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            # Load and process the data
            data = TecanDataTransformer.load_data(filepath)
            transformed = TecanDataTransformer.transform_data(data)
            well_data = TecanDataTransformer.get_transformed_data(transformed)

            # Compute growth metrics and generate plot
            plate = Wellplate((16, 24), well_data)
            unique_id = str(uuid.uuid4())
            image_filename = f'plot_{unique_id}.png'
            csv_filename = f'results_{unique_id}.tsv'
            image_path = os.path.join(RESULTS_FOLDER, image_filename)
            csv_path = os.path.join(RESULTS_FOLDER, csv_filename)

            plate.plot_raw_data(save_path=image_path)
            plate.output_csv(csv_path)

            return render_template('index.html',
                                   plot_image=f'{RESULTS_FOLDER}/{image_filename}',
                                   results_file=csv_filename)
        except Exception as e:
            return render_template('index.html', error=f'Processing error: {str(e)}')

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
