from flask import Flask, render_template, request, send_from_directory, url_for, redirect
import os
import pandas as pd
import numpy as np

from src.DataTransformer import TecanDataTransformer
from src.Wellplate import Wellplate
from src.Experiment import Experiment

from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['RESULT_FOLDER'] = RESULT_FOLDER

shared_df = pd.DataFrame()
growth_params_df = pd.DataFrame()


@app.route("/", methods=["GET", "POST"])
def index():
    global shared_df, growth_params_df

    error = None

    if request.method == "POST":
        files = request.files.getlist("csvfile")
        if not files or files == [None]:
            error = "No file(s) uploaded."
        else:
            try:
                all_dataframes = []
                all_growth_params = []

                for i, file in enumerate(files):
                    df = pd.read_csv(file)

                    df['Time [s]'] = pd.to_timedelta(df['Time']).dt.total_seconds()
                    df['Time'] = df['Time [s]']

                    transformed_data = TecanDataTransformer.transform_data(df)
                    well_data = TecanDataTransformer.get_transformed_data(transformed_data)

                    plate = Wellplate((16, 24), well_data)

                    plot_path = os.path.join(app.config['RESULT_FOLDER'], f'image_{i}.png')
                    plate.plot_raw_data(save_path=plot_path)

                    df_growth = plate.get_growth_params()

                    result_file = f"growth_results_{i}.tsv"
                    results_path = os.path.join(app.config['RESULT_FOLDER'], result_file)
                    plate.output_csv(results_path)

                    all_dataframes.append(well_data)
                    all_growth_params.append(df_growth)

                shared_df = pd.concat(all_dataframes, axis=1)
                growth_params_df = pd.concat(all_growth_params, axis=0)

                return redirect('/interactive/')

            except Exception as e:
                error = f"Processing error: {str(e)}"

    return render_template(
        "index.html",
        error=error
    )


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=True)


def moving_average(x, w=3):
    return np.convolve(x, np.ones(w) / w, mode='same')


dash_app = Dash(__name__, server=app, url_base_pathname="/interactive/")

@dash_app.server.route("/interactive/")
def dash_embed():
    return dash_app.index()


dash_app.layout = html.Div([
    html.H2("Interactive Growth Curve Viewer"),

    html.Label("Select wells (you can type to search):"),
    dcc.Dropdown(
        id="well-dropdown",
        options=[],  # populated dynamically
        placeholder="Select one or more wells",
        multi=True,
        style={
            "width": "50%",
            "marginBottom": "20px"
        }
    ),

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

    html.Div(id="hover-info", style={"marginTop": 20, "fontStyle": "italic"})
])


# Dynamically populate dropdown options after data is loaded
@dash_app.callback(
    Output("well-dropdown", "options"),
    Input("growth-graph", "id")
)
def populate_dropdown(_):
    if shared_df.empty:
        return []
    return [{"label": col, "value": col} for col in shared_df.columns if col != "Time [s]"]


@dash_app.callback(
    Output("growth-graph", "figure"),
    Output("hover-info", "children"),
    Input("well-dropdown", "value"),
    Input("smooth-toggle", "value")
)
def update_graph(selected_wells, smooth_toggle):
    if not selected_wells or shared_df.empty:
        return go.Figure(), "Select wells to display growth curves."

    show_smooth = "smooth" in smooth_toggle
    data_traces = []

    for well in selected_wells:
        if well not in shared_df.columns:
            continue

        y_data = shared_df[well].values.astype(np.float64)
        x_data = shared_df["Time [s]"].values.astype(np.float64)

        y_plot = moving_average(y_data, w=5) if show_smooth else y_data

        params = growth_params_df[growth_params_df["Well"] == well]
        if not params.empty:
            gr = params["GrowthRates"].values[0]
            tau = params["tau_values"].values[0]
            sat = params["saturate_values"].values[0]
            hover_template = (
                f"<b>{well}</b><br>"
                "Time: %{x:.1f}s<br>"
                "OD: %{y:.3f}<br>"
                f"Growth Rate: {gr:.4f}<br>"
                f"Lag Time (Tau): {tau:.2f}<br>"
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
