from flask import Flask, render_template, request, send_from_directory, url_for
import os
import pandas as pd
from src.DataTransformer import TecanDataTransformer
from src.Wellplate import Wellplate
from src.Experiment import Experiment

app = Flask(__name__)

# Folder to save results and plots
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    plot_image = None
    growth_params = None
    results_file = None

    if request.method == "POST":
        file = request.files.get("csvfile")
        if not file:
            error = "No file uploaded."
        else:
            try:
                # Read CSV
                df = pd.read_csv(file)

                # Convert 'Time' column (HH:MM:SS) to seconds for Wellplate usage
                df['Time [s]'] = pd.to_timedelta(df['Time']).dt.total_seconds()
                df['Time'] = df['Time [s]']

                # Transform data using your existing transformer
                transformed_data = TecanDataTransformer.transform_data(df)
                well_data = TecanDataTransformer.get_transformed_data(transformed_data)

                # Create Wellplate instance
                plate = Wellplate((16, 24), well_data)

                # Optionally, if you want to use Experiment with multiple plates
                # experiment = Experiment([plate], title="My Experiment")
                # experiment.plot_combined_data()

                # Plot raw data and save image
                plot_path = os.path.join(app.config['RESULT_FOLDER'], 'imageIDEA.png')
                plate.plot_raw_data(save_path=plot_path)
                plot_image = url_for('static', filename='results/imageIDEA.png')

                # Get growth parameters for display
                df_growth = plate.get_growth_params()
                growth_params = df_growth.to_dict(orient="records")

                # Save growth parameters to file
                results_file = "growth_results.tsv"
                results_path = os.path.join(app.config['RESULT_FOLDER'], results_file)
                plate.output_csv(results_path)

            except Exception as e:
                error = f"Processing error: {str(e)}"

    return render_template(
        "index.html",
        error=error,
        plot_image=plot_image,
        growth_params=growth_params,
        results_file=results_file,
    )

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
