from flask import Flask, render_template, request, send_from_directory, redirect, send_file
import os
import pandas as pd
import numpy as np
import zipfile
from io import BytesIO

from src.DataTransformer import TecanDataTransformer
from src.Wellplate import Wellplate



from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['RESULT_FOLDER'] = RESULT_FOLDER

shared_df = pd.DataFrame()
growth_params_df = pd.DataFrame()


@app.route("/", methods=["GET", "POST"])
def index():
    global shared_df, growth_params_df

    error = None
    files = []  

    if request.method == "POST":
        shared_df = pd.DataFrame() 
        growth_params_df = pd.DataFrame()

        files = request.files.getlist("csvfile")
        if not files or all(f.filename == '' for f in files):
            error = "No file(s) uploaded."
        else:
            try:
                all_dataframes = []
                all_growth_params = []
                time_column = None 

                for i, file in enumerate(files):
                    if file.filename == '':
                        continue
                        
                    df = pd.read_csv(file)

                    # Convert 'Time' column to seconds
                    df['Time [s]'] = pd.to_timedelta(df['Time']).dt.total_seconds()
                    df['Time'] = df['Time [s]']

                    transformed_data = TecanDataTransformer.transform_data(df)
                    well_data = TecanDataTransformer.get_transformed_data(transformed_data)
                    
                    # Store the time column from the first file
                    if time_column is None:
                        time_column = well_data['Time [s]'].copy()
                    
                    # Remove Time column before renaming to avoid duplicates
                    well_cols = [col for col in well_data.columns if col != 'Time [s]']
                    well_data_no_time = well_data[well_cols].copy()
                    
                    renamed_cols = {col: f"File{i+1}_{col}" for col in well_cols}
                    well_data_no_time = well_data_no_time.rename(columns=renamed_cols)

                    plate_shape = Wellplate.detect_plate_layout(well_data.columns)
                    print(f"Detected plate layout: {plate_shape}")
                    plate = Wellplate(plate_shape, well_data)

                    df_growth = plate.get_growth_params()

                    result_file = f"growth_results_{i+1}.tsv"
                    results_path = os.path.join(app.config['RESULT_FOLDER'], result_file)
                    plate.output_csv(results_path)

                    all_dataframes.append(well_data_no_time)
                    all_growth_params.append(df_growth)

                if all_dataframes:  # Only proceed if we have data
                    # Concatenate well dataframes side by side (without time columns)
                    shared_df = pd.concat(all_dataframes, axis=1)
                    # Add back the time column
                    shared_df['Time [s]'] = time_column

                    # Concatenate growth params vertically
                    growth_params_df = pd.concat(all_growth_params, axis=0, ignore_index=True)

                    # Save batch summary CSV
                    growth_params_df.to_csv(
                        os.path.join(app.config['RESULT_FOLDER'], "batch_summary.tsv"),
                        sep="\t", index=False
                    )

                    return redirect('/interactive/')
                else:
                    error = "No valid files processed."

            except Exception as e:
                error = f"Processing error: {str(e)}"
                print(f"Debug - Error details: {e}")

    return render_template(
        "index.html",
        error=error
    )
@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=True)


@app.route("/download/batch_summary.tsv")
def download_summary():
    return send_from_directory(app.config["RESULT_FOLDER"], "batch_summary.tsv", as_attachment=True)


@app.route("/download/plots.png")
def download_plots():
    global shared_df

    if shared_df.empty:
        return "No data loaded. Please upload data first.", 400

    df_with_time = shared_df.copy()
    plate = Wellplate((16, 24), df_with_time, well_plate_name='current')

    plots_dir = os.path.join(app.config['RESULT_FOLDER'])
    os.makedirs(plots_dir, exist_ok=True)

    save_path = os.path.join(plots_dir, "plots.png")

    # Generate ONE combined plot of all wells and save as PNG
    plate.plot_raw_data(save_path=save_path)

    # Send the combined PNG file
    return send_file(
        save_path,
        mimetype='image/png',
        download_name='plots.png',
        as_attachment=True
    )


def moving_average(x, w=3):
    return np.convolve(x, np.ones(w) / w, mode='same')


dash_app = Dash(__name__, server=app, url_base_pathname="/interactive/")



dash_app.layout = html.Div([
    html.H2("Interactive Growth Curve Viewer"),

    html.Div([
        html.A(
            "Return to Upload Page",
            href="/",
            style={
                "display": "inline-block",
                "padding": "8px 16px",
                "backgroundColor": "#ffffff",
                "color": "Black",
                "border": "1px solid #007bff",
                "textDecoration": "none",
                "borderRadius": "4px",
                "marginBottom": "20px",
                "fontWeight": "bold"
            }
        )
    ]),


    html.Label("Select wells (you can type to search):", style={"marginBottom": "10px", "display": "block"}),
    

    html.Div([
        dcc.Dropdown(
            id="well-dropdown-1",
            options=[],
            placeholder="Select wells from File 1",
            multi=True,
            style={"width": "100%"}
        ),
    ], style={"width": "45%", "display": "inline-block", "marginRight": "5%", "verticalAlign": "top"}),

    html.Div([
        dcc.Dropdown(
            id="well-dropdown-2",
            options=[],
            placeholder="Select wells from File 2",
            multi=True,
            style={"width": "100%"}
        ),
    ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top"}),

    dcc.Checklist(
        id="smooth-toggle",
        options=[{"label": "Smooth Curve", "value": "smooth"}],
        value=[],
        style={"marginBottom": "20px"}
    ),

    dcc.Graph(
        id="growth-graph",
        config={"modeBarButtonsToAdd": ["downloadImage"]},
        style={"width": "90vw", "height": "500px", "margin": "0 auto"}
    ),

    html.Div(id="hover-info", style={"marginTop": 20, "fontStyle": "italic"}),

    html.Br(),

    html.Div([
        html.A("Download Batch Summary (.tsv)", href="/download/batch_summary.tsv", target="_blank"),
        html.Br(),
        html.A(" Download Well Plots (.png)", href="/download/plots.png", target="_blank"),
    ], className="download-buttons"),


])



@dash_app.callback(
    [Output("well-dropdown-1", "options"),
     Output("well-dropdown-2", "options")],
    [Input("well-dropdown-1", "id")]
)
def populate_dropdowns(_):
    try:
        if shared_df.empty:
            return [], []
        
        print(f"Debug - DataFrame columns: {list(shared_df.columns)}")  # Debug line
        
        options_1 = [{"label": col.replace("File1_", ""), "value": col} 
                     for col in shared_df.columns if col.startswith("File1_")]
        options_2 = [{"label": col.replace("File2_", ""), "value": col} 
                     for col in shared_df.columns if col.startswith("File2_")]
        
        print(f"Debug - Options 1: {len(options_1)}, Options 2: {len(options_2)}")  # Debug line
        
        return options_1, options_2
    except Exception as e:
        print(f"Debug - Dropdown error: {e}")
        return [], []


@dash_app.callback(
    [Output("growth-graph", "figure"),
     Output("hover-info", "children")],
    [Input("well-dropdown-1", "value"),
     Input("well-dropdown-2", "value"),
     Input("smooth-toggle", "value")]
)
def update_graph(selected_wells_1, selected_wells_2, smooth_toggle):
    if shared_df.empty:
        return go.Figure(), "No data loaded. Please upload files first."
    
    if not selected_wells_1 and not selected_wells_2:
        return go.Figure(), "Select wells to display growth curves."

    show_smooth = "smooth" in (smooth_toggle or [])
    data_traces = []

    # Combine wells from both dropdowns
    selected_wells = (selected_wells_1 or []) + (selected_wells_2 or [])

    for well in selected_wells:
        if well not in shared_df.columns:
            continue

        y_data = shared_df[well].values.astype(np.float64)
        x_data = shared_df["Time [s]"].values.astype(np.float64)

        y_plot = moving_average(y_data, w=5) if show_smooth else y_data

        # Extract the original well name for lookup (remove File1_ or File2_ prefix)
        original_well = well.split('_', 1)[1] if '_' in well else well
        params = growth_params_df[growth_params_df["Well"] == original_well]
        
        if not params.empty:
            gr = params["GrowthRates"].values[0]
            tau = params["tau_values"].values[0]
            sat = params["saturate_values"].values[0]
            hover_template = (
                f"<b>{well}</b><br>"
                "Time: %{x:.1f}s<br>"
                "OD: %{y:.3f}<br>"
                f"Growth Rate: {gr:.4f}<br>"
                f"Lag Time (Tau): {tau:.2e}<br>"
                f"Saturation OD: {sat:.3f}<extra></extra>"
            )
        else:
            hover_template = (
                f"<b>{well}</b><br>"
                "Time: %{x:.1f}s<br>"
                "OD: %{y:.3f}<extra></extra>"
            )

        trace = go.Scatter(
            x=x_data,
            y=y_plot,
            mode="lines+markers",
            name=well,
            hovertemplate=hover_template
        )
        data_traces.append(trace)

    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title="Growth Curves",
        xaxis_title="Time (seconds)",
        yaxis_title="Optical Density (OD)",
        hovermode="closest"
    )

    hover_info_text = f"Displaying {len(data_traces)} wells."
    return fig, hover_info_text

if __name__ == "__main__":
    app.run(debug=True)
