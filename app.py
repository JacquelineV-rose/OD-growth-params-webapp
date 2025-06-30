from flask import Flask, request, render_template, send_from_directory, url_for
import pandas as pd
import os

from src.DataTransformer import TecanDataTransformer
from src.Wellplate import Wellplate

app = Flask(__name__)
app.config["RESULT_FOLDER"] = "static/results"
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    plot_image = None
    growth_params = None
    results_file = None

    if request.method == "POST":
        file = request.files.get("csvfile")
        if not file:
            error = "No file uploaded"
        else:
            try:
                # Read CSV from uploaded file directly
                df = pd.read_csv(file, sep=",")  # or sep="\t" if needed

                # Use transformer to clean/transform df
                transformer = TecanDataTransformer()
                transformed_df = transformer.transform_df(df)

                # Pass transformed data to Wellplate
                plate = Wellplate(layout=(8, 12), well_data=transformed_df)
                plate.compute_params()

                # Save plot and output file
                plot_path = os.path.join(app.config["RESULT_FOLDER"], "imageIDEA.png")
                plate.plot_raw_data(save_path=plot_path)

                results_file = "results.tsv"
                results_path = os.path.join(app.config["RESULT_FOLDER"], results_file)
                plate.output_csv(results_path)

                plot_image = url_for("static", filename="results/imageIDEA.png")
                growth_params = plate.get_growth_params().to_dict(orient="records")

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
