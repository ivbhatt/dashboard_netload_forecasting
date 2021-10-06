import os, sys
from dash.dcc.Checklist import Checklist
from dash.dependencies import Input, Output

import numpy as np
from numpy.lib.npyio import load
import pandas as pd

import dash
from dash import html, dcc
from pandas.core.frame import DataFrame
import plotly.express as px

from utils import print_info, print_warn


app = dash.Dash(__name__)


DATA_PATH = os.path.join("..", "data")

datasets = os.listdir(DATA_PATH)

if len(datasets) != 8:
    print_warn("All 8 datasets not found." )

print_info("Datasets:", datasets)



def load_dataset(dataset):
    dataFrame = pd.DataFrame(columns=["Dataset", "Location", "Year", "Month", "Day", "Weekday", "Hour","Demand"])

    for root, dirs, files in os.walk(os.path.join(DATA_PATH, dataset, "data_cleaned")):
        print_info("Working on dataset:", dataset)
        if "Date.csv" in files:
            date = pd.read_csv(os.path.join(root, "Date.csv"))

            month_series = date["Month"]

            curr_year = 1
            year_series = [0 for i in month_series]

            for i in range(len(month_series)-1):
                if month_series[i+1] >= month_series[i]:
                    year_series[i] += curr_year
                else:
                    year_series[i] += curr_year
                    curr_year+=1

            year_series[-1] += curr_year

            for file in files:
                if file != "Date.csv":
                    print_info("Working on dataset:", dataset," location:", file.split(".")[0])

                    t = pd.read_csv(os.path.join(root, file))
                    
                    if "Demand" in t.columns:
                        temp_df = pd.DataFrame(data = t["Demand"])
                    else:
                        temp_df = pd.DataFrame(data = t["Net"])
                        temp_df["Demand"] = temp_df["Net"]
                        temp_df.drop(columns=["Net"])


                    temp_df["Dataset"] = dataset
                    temp_df["Location"] = file.split(".")[0]
                    temp_df["Year"] = year_series
                    temp_df["Month"] = month_series
                    temp_df["Day"] = date["Day"]
                    temp_df["Hour"] = date["Hour"]
                    temp_df["Weekday"] = date["Weekday"]
                        

                    dataFrame = pd.concat([dataFrame, temp_df])

        else:
            temp_df = pd.read_csv(os.path.join(root, files[0]))[["Month","Day","Weekday","Hour","Demand"]]
            temp_df["Dataset"] = dataset
            temp_df["Location"] = "-"
            temp_df["Year"] = "1"

            dataFrame = pd.concat([dataFrame, temp_df])
    return dataFrame

dataFrames = {}
for dataset in datasets:
    dataFrames[dataset] = load_dataset(dataset)

dataFrame = dataFrames["S1"]

hour_demand = px.box(dataFrame[["Hour", "Demand", "Weekday"]], y = "Demand", x = "Hour", color = "Weekday")
month_demand = px.box(dataFrame[["Month", "Demand",  "Weekday"]], y = "Demand", x = "Month", color = "Weekday")
year_demand = px.box(dataFrame[["Year", "Demand",  "Weekday"]], y = "Demand", x = "Year", color = "Weekday")

heatmap = px.imshow(np.array(dataFrame.groupby(["Month", "Hour"])["Demand"].mean()).reshape(12, 24), color_continuous_scale="icefire")

app.layout = html.Div(children=[
    html.H1(id = "heading", children="Solar power dashboard"),
    html.P(id ="active_dataset", children="S1"),
    dcc.RadioItems(id = "data_selector", options = [
        {"label":"S1", "value":"S1"},
        {"label":"S2", "value":"S2"},
        {"label":"S3", "value":"S3"},
        {"label":"S4", "value":"S4"},
        {"label":"L1", "value":"L1"},
        {"label":"L2", "value":"L2"},
        {"label":"L3", "value":"L3"},
        {"label":"L4", "value":"L4"},
    ], value = "S1"
    ),
    dcc.Graph(id = "hour_demand", figure=hour_demand),
    dcc.Graph(id = "month_demand", figure=month_demand),
    dcc.Graph(id = "year_demand", figure=year_demand),
    dcc.Graph(id = "heatmap",figure = heatmap)
])

@app.callback(
    Output(component_id="active_dataset", component_property="children"),
    Input(component_id="data_selector", component_property="value"))
def load_new_datasset(selected_dataset):
    global dataFrame
    dataFrame = dataFrames[selected_dataset]
    return selected_dataset

@app.callback(
    Output(component_id="hour_demand", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_dataset(selected_dataset):
    # filtered_dataFrame = load_dataset(selected_dataset)

    hour_demand = px.box(dataFrame[["Hour", "Demand", "Weekday"]], y = "Demand", x = "Hour", color = "Weekday")

    hour_demand.update_layout()

    return hour_demand


@app.callback(
    Output(component_id="month_demand", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_dataset(selected_dataset):
    # filtered_dataFrame = load_dataset(selected_dataset)

    month_demand = px.box(dataFrame[["Month", "Demand", "Weekday"]], y = "Demand", x = "Month", color = "Weekday")

    month_demand.update_layout()

    return month_demand

@app.callback(
    Output(component_id="year_demand", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_dataset(selected_dataset):
    # filtered_dataFrame = load_dataset(selected_dataset)
    year_demand = px.box(dataFrame[["Year", "Demand", "Weekday"]], y = "Demand", x = "Year", color = "Weekday")
    year_demand.update_layout()

    return year_demand

@app.callback(
    Output(component_id="heatmap", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_dataset(selected_dataset):

    heatmap = px.imshow(np.array(dataFrame.groupby(["Month", "Hour"])["Demand"].mean()).reshape(12, 24), color_continuous_scale="icefire")


    heatmap.update_layout()

    return heatmap



if __name__ == "__main__":
    app.run_server(debug=True)