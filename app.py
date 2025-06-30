import os
import uuid
from flask import Flask, request, render_template, send_from_directory
from src.DataTransformer import TecanDataTransformer
from src.Wellplate import Wellplate

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('csvfile')
        if not file or file.filename == '':
            return render_template('index.html', error='No file selected.')

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            # Load and transform data
            raw_data = TecanDataTransformer.load_data(filepath)
            transformed_data = TecanDataTransformer.transform_data(raw_data)
            well_data = TecanDataTransformer.get_transformed_data(transformed_data)

            # Create Wellplate instance (16 rows, 24 cols)
            plate = Wellplate((16,24), well_data)
            plate.compute_params()

            # Save plot image
            unique_id = str(uuid.uuid4())
            plot_path = os.path.join(RESULTS_FOLDER, f'growth_plot_{unique_id}.png')
            plate.plot_raw_data(save_path=plot_path)

            # Save growth params CSV
            csv_path = os.path.join(RESULTS_FOLDER, f'growth_params_{unique_id}.tsv')
            plate.output_csv(csv_path)

            # Pass params as list of dicts to template
            growth_params = plate.get_growth_params().to_dict(orient='records')

            return render_template('index.html',
                                   plot_image=f'{RESULTS_FOLDER}/growth_plot_{unique_id}.png',
                                   results_file=f'growth_params_{unique_id}.tsv',
                                   growth_params=growth_params)

        except Exception as e:
            return render_template('index.html', error=f'Processing error: {str(e)}')

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
