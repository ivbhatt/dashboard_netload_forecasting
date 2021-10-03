import os, sys
from dash.dcc.Checklist import Checklist
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd

import dash
from dash import html, dcc
import plotly.express as px

app = dash.Dash(__name__)

S1_PATH = os.path.join("..","data","S1","data_cleaned", "data.csv")
S2_PATH = os.path.join("..","data","S2","data_cleaned", "data.csv")



s1 = pd.read_csv(S1_PATH)
s1["dataset"] = "s1"

s2 = pd.read_csv(S2_PATH)
s2["dataset"] = "s2"

s = pd.concat([s1, s2])

hour_demand = px.box(s[["Hour", "Demand", "Weekday"]], y = "Demand", x = "Hour", color = "Weekday")
month_demand = px.box(s[["Month", "Demand",  "Weekday"]], y = "Demand", x = "Month", color = "Weekday")


heatmap = px.imshow(np.array(s.groupby(["Month", "Hour"])["Demand"].mean()).reshape(12, 24), color_continuous_scale="icefire")

app.layout = html.Div(children=[
    html.H1(id = "heading", children="Solar power dashboard"),
    dcc.RadioItems(id = "data_selector", options = [
        {"label":"s1", "value":"s1"},
        {"label":"s2", "value":"s2"},
    ], value = "s1"
    ),
    dcc.Graph(id = "hour_demand", figure=hour_demand),
    dcc.Graph(id = "month_demand", figure=month_demand),
    dcc.Graph(id = "heatmap",figure = heatmap)
])

@app.callback(
    Output(component_id="hour_demand", component_property="figure"),
    Input(component_id="data_selector", component_property="value"))
def update_dataset(selected_dataset):
    filtered_s = s[s.dataset == selected_dataset]

    hour_demand = px.box(filtered_s[["Hour", "Demand", "Weekday"]], y = "Demand", x = "Hour", color = "Weekday")

    hour_demand.update_layout(transition_duration = 500)

    return hour_demand


@app.callback(
    Output(component_id="month_demand", component_property="figure"),
    Input(component_id="data_selector", component_property="value"))
def update_dataset(selected_dataset):
    filtered_s = s[s.dataset == selected_dataset]

    month_demand = px.box(filtered_s[["Month", "Demand", "Weekday"]], y = "Demand", x = "Month", color = "Weekday")

    month_demand.update_layout(transition_duration = 500)

    return month_demand


@app.callback(
    Output(component_id="heatmap", component_property="figure"),
    Input(component_id="data_selector", component_property="value"))
def update_dataset(selected_dataset):
    filtered_s = s[s.dataset == selected_dataset]

    heatmap = px.imshow(np.array(filtered_s.groupby(["Month", "Hour"])["Demand"].mean()).reshape(12, 24), color_continuous_scale="icefire")


    heatmap.update_layout(transition_duration = 500)

    return heatmap



if __name__ == "__main__":
    app.run_server(debug=True)