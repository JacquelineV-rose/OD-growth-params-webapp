from flask import Flask, render_template, request, send_from_directory, redirect, send_file
import os
import pandas as pd
import numpy as np
import time

from src.DataTransformer import TecanDataTransformer
from src.Wellplate import Wellplate

from dash import Dash, html, dcc, Input, Output, State, callback_context
from dash.dependencies import ALL
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
num_files = 0
data_timestamp = 0  # Add timestamp to track when data changes

@app.route("/", methods=["GET", "POST"])
def index():
    global shared_df, growth_params_df, num_files, data_timestamp

    error = None
    files = []

    if request.method == "POST":
        # Reset global variables
        shared_df = pd.DataFrame()
        growth_params_df = pd.DataFrame()
        num_files = 0
        data_timestamp = time.time()  # Update timestamp

        files = request.files.getlist("csvfile")
        if not files or all(f.filename == '' for f in files):
            error = "No file(s) uploaded."
        else:
            try:
                all_dataframes = []
                all_growth_params = []
                time_column = None
                valid_file_count = 0

                for i, file in enumerate(files):
                    if file.filename == '':
                        continue

                    df = pd.read_csv(file)

                    df['Time [s]'] = pd.to_timedelta(df['Time']).dt.total_seconds()
                    df['Time'] = df['Time [s]']

                    transformed_data = TecanDataTransformer.transform_data(df)
                    well_data = TecanDataTransformer.get_transformed_data(transformed_data)

                    if time_column is None:
                        time_column = well_data['Time [s]'].copy()

                    well_cols = [col for col in well_data.columns if col != 'Time [s]']
                    well_data_no_time = well_data[well_cols].copy()

                    valid_file_count += 1
                    renamed_cols = {col: f"File{valid_file_count}_{col}" for col in well_cols}
                    well_data_no_time = well_data_no_time.rename(columns=renamed_cols)

                    plate_shape = Wellplate.detect_plate_layout(well_data.columns)
                    plate = Wellplate(plate_shape, well_data)

                    df_growth = plate.get_growth_params()
                    df_growth['File_Number'] = valid_file_count

                    result_file = f"growth_results_{valid_file_count}.tsv"
                    results_path = os.path.join(app.config['RESULT_FOLDER'], result_file)
                    plate.output_csv(results_path)

                    all_dataframes.append(well_data_no_time)
                    all_growth_params.append(df_growth)

                if all_dataframes:
                    num_files = valid_file_count

                    shared_df = pd.concat(all_dataframes, axis=1)
                    shared_df['Time [s]'] = time_column

                    growth_params_df = pd.concat(all_growth_params, axis=0, ignore_index=True)

                    growth_params_df.to_csv(
                        os.path.join(app.config['RESULT_FOLDER'], "batch_summary.tsv"),
                        sep="\t", index=False
                    )

                    data_timestamp = time.time()
                    return redirect('/interactive/')
                else:
                    error = "No valid files processed."

            except Exception as e:
                error = f"Processing error: {str(e)}"
                import traceback
                traceback.print_exc()

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
    plate.plot_raw_data(save_path=save_path)

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


    dcc.Store(id="data-store"),

    html.Div(id="status-div", style={"marginBottom": "20px"}),

    html.Div(id="dropdown-container"),

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
        html.A("Download Well Plots (.png)", href="/download/plots.png", target="_blank"),
    ], className="download-buttons"),
])

@dash_app.callback(
    Output("data-store", "data"),
    [Input("dropdown-container", "id")] 
)
def update_data_store(_):
    global num_files, shared_df, data_timestamp

    return {
        "num_files": num_files,
        "has_data": not shared_df.empty,
        "timestamp": data_timestamp,
        "columns": list(shared_df.columns) if not shared_df.empty else []
    }

@dash_app.callback(
    [Output("dropdown-container", "children"),
     Output("status-div", "children")],
    [Input("data-store", "data")]
)
def update_dropdown_container(data_state):
    if not data_state or not data_state.get("has_data", False):
        status = html.Div([
            html.P("No data loaded. Please upload files first.",
                   style={"color": "#666", "fontStyle": "italic"})
        ])
        return [], status

    num_files_current = data_state.get("num_files", 0)
    columns = data_state.get("columns", [])

    status = html.Div([
        html.P(f"✓ Data loaded: {num_files_current} files",
               style={"color": "#28a745", "fontWeight": "bold"})
    ])

    dropdowns = []
    for i in range(1, num_files_current + 1):
        file_columns = [col for col in columns if col.startswith(f"File{i}_")]
        options = [{"label": col.replace(f"File{i}_", ""), "value": col} for col in file_columns]

        dropdowns.append(
            dcc.Dropdown(
                id={'type': 'well-dropdown', 'index': i},  
                options=options,
                placeholder=f"Select wells from File {i}",
                multi=True,
                style={"width": "48%", "display": "inline-block", "marginRight": "2%"}
            )
        )

    return dropdowns, status

@dash_app.callback(
    [Output("growth-graph", "figure"),
     Output("hover-info", "children")],
    [Input({'type': 'well-dropdown', 'index': ALL}, 'value'),
     Input("smooth-toggle", "value")],
    prevent_initial_call=True
)
def update_graph(selected_wells_lists, smooth_toggle):
    global shared_df, growth_params_df, num_files

    if shared_df.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data loaded. Please upload files first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return empty_fig, "No data loaded."

    selected_wells = []
    for wells in selected_wells_lists:
        if wells:
            selected_wells.extend(wells)

    if not selected_wells:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Select wells to display growth curves.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return empty_fig, "Select wells."

    show_smooth = "smooth" in (smooth_toggle or [])
    data_traces = []

    for well in selected_wells:
        x_data = shared_df["Time [s]"].values
        y_data = shared_df[well].values
        y_plot = moving_average(y_data, w=5) if show_smooth else y_data

        trace = go.Scatter(
            x=x_data, y=y_plot, mode="lines+markers", name=well
        )
        data_traces.append(trace)

    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title="Growth Curves",
        xaxis_title="Time (seconds)",
        yaxis_title="OD600",
        hovermode="closest"
    )

    return fig, f"Displaying {len(data_traces)} wells."

if __name__ == "__main__":
    app.run(debug=True)
