import os, sys
from dash.dcc.Checklist import Checklist
from dash.dcc.RadioItems import RadioItems
from dash.dependencies import Input, Output, State
from dash.html.Label import Label

import numpy as np
from numpy.lib.npyio import load
import pandas as pd

import dash
from dash import html, dcc
from pandas.core.algorithms import unique
from pandas.core.frame import DataFrame
import plotly.graph_objects as go
import plotly.express as px

from utils import print_info, print_warn, convert_to_message, normalize

current_selection = {
    "dataset" : "S1",
    "locations" : ["ALL"],
    "years" : ["ALL"]
}

app = dash.Dash(__name__)

DATA_PATH = os.path.join("..", "data")

datasets = os.listdir(DATA_PATH)
if len(datasets) != 8:
    print_warn("All 8 datasets not found." )
print_info("Datasets:", datasets)



def load_dataset(dataset):
    dataFrame = pd.DataFrame(columns=["Dataset", "Location", "Year", "Month", "Day", "Weekday", "Hour","Demand", "Temperature"])

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
                    temp_df["Temperature"] = t["Temperature"]
                    

                    dataFrame = pd.concat([dataFrame, temp_df])

        else:
            temp_df = pd.read_csv(os.path.join(root, files[0]))[["Month","Day","Weekday","Hour","Demand", "Temperature"]]
            temp_df["Dataset"] = dataset
            temp_df["Location"] = "ALL"
            temp_df["Year"] = "ALL"

            dataFrame = pd.concat([dataFrame, temp_df])
    return dataFrame

dataFrames = {}
for dataset in ["S1", "S4"]:
# for dataset in datasets:
    dataFrames[dataset] = load_dataset(dataset)


dataFrame = dataFrames["S1"]

## HOUR BREAKUP
summarized = dataFrame.groupby(["Weekday", "Hour"])["Demand"].describe()

hour_demand = go.Figure()
hour_demand.add_trace(go.Box(q1 = summarized["25%"][1], q3 = summarized["75%"][1],
                             median = summarized["50%"][1], name = "Weekday",
                             lowerfence= summarized["min"][1], upperfence= summarized["max"][1]
 ))
hour_demand.add_trace(go.Box(q1 = summarized["25%"][0], q3 = summarized["75%"][0],
                             median = summarized["50%"][0], name = "Weekend",
                             lowerfence= summarized["min"][0], upperfence= summarized["max"][0]
 ))

hour_demand.update_layout(boxmode="group")
##

## MONTH BREAKUP
summarized = dataFrame.groupby(["Weekday", "Month"])["Demand"].describe()

month_demand = go.Figure()
month_demand.add_trace(go.Box(q1 = summarized["25%"][1], q3 = summarized["75%"][1],
                             median = summarized["50%"][1], name = "Weekday",
                             lowerfence= summarized["min"][1], upperfence= summarized["max"][1]
 ))
month_demand.add_trace(go.Box(q1 = summarized["25%"][0], q3 = summarized["75%"][0],
                             median = summarized["50%"][0], name = "Weekend",
                             lowerfence= summarized["min"][0], upperfence= summarized["max"][0]
 ))
month_demand.update_layout(boxmode="group")
##

## YEAR BREAKUP
summarized = dataFrame.groupby(["Weekday", "Year"])["Demand"].describe()

year_demand = go.Figure()
year_demand.add_trace(go.Box(q1 = summarized["25%"][1], q3 = summarized["75%"][1],
                             median = summarized["50%"][1], name = "Weekday",
                             lowerfence= summarized["min"][1], upperfence= summarized["max"][1]
 ))
year_demand.add_trace(go.Box(q1 = summarized["25%"][0], q3 = summarized["75%"][0],
                             median = summarized["50%"][0], name = "Weekend",
                             lowerfence= summarized["min"][0], upperfence= summarized["max"][0]
 ))
year_demand.update_layout(boxmode="group")
##

# heatmaps
month_hour_heatmap_data = np.array(dataFrame.groupby(["Month", "Hour"])["Demand"].median()).reshape(12,24)
month_hour_heatmap = px.imshow(normalize(month_hour_heatmap_data), color_continuous_scale="Bluered")
location_correlation = px.imshow(np.array([1]).reshape(1,1), color_continuous_scale="Bluered")

## line chart
autocorrelation = go.Figure()

## scatter plot
temp_demand_correlation = go.Figure()

app.layout = html.Div(children=[
    html.H1(id = "heading", children="Solar power dashboard"),
    html.P(id ="active_dataset", children=convert_to_message(current_selection)),
    html.Div(id = "selector", children = [
        html.Label(children = "Select the dataset:"),
        dcc.RadioItems(id = "data_selector", options = [
            {"label":"S1", "value":"S1"},
            {"label":"S2", "value":"S2"},
            {"label":"S3", "value":"S3"},
            {"label":"S4", "value":"S4"},
            {"label":"L1", "value":"L1"},
            {"label":"L2", "value":"L2"},
            {"label":"L3", "value":"L3"},
            {"label":"L4", "value":"L4"},
        ], value = "S1"),
        html.Label(children = "Select the Location:"),
        dcc.Checklist( id = "location_selector", options = [
            {"label": "ALL", "value":"ALL"}
        ], value=["ALL"]),
        html.Label(children = "Select the Years:"),
        dcc.Checklist( id = "year_selector", options = [
            {"label": "ALL", "value":"ALL"}
        ], value=["ALL"]),
        html.Button(id = "submit", type = "submit", children = "Refresh!")
    ], style = {"border" : "1px solid black"}
    ),
    dcc.Graph(id = "hour_demand", figure=hour_demand),
    dcc.Graph(id = "month_demand", figure=month_demand),
    dcc.Graph(id = "year_demand", figure=year_demand),
    dcc.Graph(id = "month_hour_heatmap",figure = month_hour_heatmap),
    dcc.Graph(id = "location_correlation",figure = location_correlation),
    dcc.Graph(id = "autocorrelation", figure = autocorrelation),
    dcc.Graph(id = "temp_demand_correlation", figure = temp_demand_correlation)
])


@app.callback(
    output = {
        "location_selector_options":Output(component_id="location_selector", component_property="options"),
        "location_selector_value":Output(component_id="location_selector", component_property="value"),
        "year_selector_options":Output(component_id="year_selector", component_property="options"),
        "year_selector_value":Output(component_id="year_selector", component_property="value"),
        "active_dataset":Output(component_id="active_dataset", component_property="children")
    },
    inputs ={
        "button":Input(component_id="submit", component_property="n_clicks")
    },
    state = {
        "selected_locations":State(component_id="location_selector", component_property="value"),
        "selected_dataset":State(component_id="data_selector", component_property="value"),
        "selected_years":State(component_id="year_selector", component_property="value"),
    })


def load_new_datasset(button, selected_dataset, selected_locations, selected_years):
    print(button)
    global dataFrame
    global current_selection

    # update form fields
    # update global current_selection
    # update dataFrame
    # update active_dataset hence triggerring graph updates

    if current_selection["dataset"] != selected_dataset:
        
        dataFrame = dataFrames[selected_dataset]

        current_selection["dataset"] = selected_dataset
        current_selection["locations"] = [i for i in dataFrame.Location.unique()]
        current_selection["years"] = [i for i in dataFrame.Year.unique()]

        returnLocations = []
        for location in current_selection["locations"]:
            returnLocations.append({"label":location, "value":location})
        returnLocation = current_selection["locations"]

        returnYears = []
        for year in current_selection["years"]:
            returnYears.append({"label":year, "value":year})
        returnYear = current_selection["years"]

        return {
            "location_selector_options": returnLocations,
            "location_selector_value": returnLocation,
            "year_selector_options": returnYears,
            "year_selector_value": returnYear,
            "active_dataset": convert_to_message(current_selection)
        }
    else:
        returnLocations = []
        for location in dataFrames[current_selection["dataset"]].Location.unique():
            returnLocations.append({"label":location, "value":location})

        returnYears = []
        for year in dataFrames[current_selection["dataset"]].Year.unique():
            returnYears.append({"label":year, "value":year})

        current_selection["locations"] = selected_locations
        current_selection["years"] = selected_years


        print(current_selection)
        print(dataFrame.head())

        dataFrame = dataFrames[current_selection["dataset"]]
        print(dataFrame.shape)
        
        dataFrame = dataFrame.loc[dataFrame.Year.isin(current_selection["years"])]
        dataFrame = dataFrame.loc[dataFrame.Location.isin(current_selection["locations"])]
        
        print(dataFrame.shape)
        
        return {
            "location_selector_options": returnLocations,
            "location_selector_value": current_selection["locations"],
            "year_selector_options": returnYears,
            "year_selector_value": current_selection["years"],
            "active_dataset": convert_to_message(current_selection)
        }


    return current_selection


@app.callback(
    Output(component_id="hour_demand", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_dataset(selected_dataset):
    summarized = dataFrame.groupby(["Weekday", "Hour"])["Demand"].describe()
    hour_demand = go.Figure()
    hour_demand.add_trace(go.Box(q1 = summarized["25%"][1], q3 = summarized["75%"][1],
                                median = summarized["50%"][1], name = "Weekday",
                                lowerfence= summarized["min"][1], upperfence= summarized["max"][1]
    ))
    hour_demand.add_trace(go.Box(q1 = summarized["25%"][0], q3 = summarized["75%"][0],
                                median = summarized["50%"][0], name = "Weekend",
                                lowerfence= summarized["min"][0], upperfence= summarized["max"][0]
    ))

    hour_demand.update_layout(boxmode="group")

    return hour_demand

@app.callback(
    Output(component_id="month_demand", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_dataset(selected_dataset):
    summarized = dataFrame.groupby(["Weekday", "Month"])["Demand"].describe()

    month_demand = go.Figure()
    month_demand.add_trace(go.Box(q1 = summarized["25%"][1], q3 = summarized["75%"][1],
                                median = summarized["50%"][1], name = "Weekday",
                                lowerfence= summarized["min"][1], upperfence= summarized["max"][1]
    ))
    month_demand.add_trace(go.Box(q1 = summarized["25%"][0], q3 = summarized["75%"][0],
                                median = summarized["50%"][0], name = "Weekend",
                                lowerfence= summarized["min"][0], upperfence= summarized["max"][0]
    ))

    month_demand.update_layout(boxmode="group")
    return month_demand

@app.callback(
    Output(component_id="year_demand", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_dataset(selected_dataset):
    summarized = dataFrame.groupby(["Weekday", "Year"])["Demand"].describe()

    year_demand = go.Figure()
    year_demand.add_trace(go.Box(q1 = summarized["25%"][1], q3 = summarized["75%"][1],
                                median = summarized["50%"][1], name = "Weekday",
                                lowerfence= summarized["min"][1], upperfence= summarized["max"][1]
    ))
    year_demand.add_trace(go.Box(q1 = summarized["25%"][0], q3 = summarized["75%"][0],
                                median = summarized["50%"][0], name = "Weekend",
                                lowerfence= summarized["min"][0], upperfence= summarized["max"][0]
    ))

    year_demand.update_layout(boxmode="group")
    return year_demand

@app.callback(
    Output(component_id="month_hour_heatmap", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_dataset(selected_dataset):
    month_hour_heatmap_data = np.array(dataFrame.groupby(["Month", "Hour"])["Demand"].median()).reshape(12,24)
    month_hour_heatmap = px.imshow(normalize(month_hour_heatmap_data), color_continuous_scale="Bluered")
    month_hour_heatmap.update_layout()
    return month_hour_heatmap

@app.callback(
    Output(component_id="location_correlation", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_dataset(selected_dataset):

    locations = dataFrame.Location.unique()
    print (locations)

    corrs = [[ 0 for i in range(len(locations)) ] for k in range(len(locations))]

    for i in range(len(locations)):
        for j in range(i, len(locations)):

            a = dataFrame.loc[dataFrame.Location.isin([locations[i]])].sort_values(by = ["Year", "Month", "Day", "Hour"])["Demand"]
            b = dataFrame.loc[dataFrame.Location.isin([locations[j]])].sort_values(by = ["Year", "Month", "Day", "Hour"])["Demand"]
            
            t = np.corrcoef(a, b)
            
            if len(t) == 1:
                corrs[i][j] = t
            else:
                corrs[i][j] = t[0][1]

            corrs[j][i] = corrs[i][j]

    location_correlation = px.imshow(corrs, color_continuous_scale="Bluered")
    return location_correlation


@app.callback(
    Output(component_id="autocorrelation", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_dataset(selected_dataset):
    locations = dataFrame.Location.unique()
    temp = go.Figure()

    corrs = pd.DataFrame()

    
    for i in range(len(locations)):
        a = dataFrame.loc[dataFrame.Location.isin([locations[i]])].sort_values(by = ["Year", "Month", "Day", "Hour"])
        
        a = a["Demand"]

        t = np.correlate(a, a, mode = "full")
        corrs[i] = t[len(t)//2: 201 + len(t)//2][:]
        corrs[i] /= max(corrs[i])
 
        temp = temp.add_trace(go.Scatter(y = corrs[i], x = [i for i in range(201)],
            mode = "lines+markers", name = locations[i]))

    return temp


@app.callback(
    Output(component_id="temp_demand_correlation", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_dataset(selected_dataset):
    locations = dataFrame.Location.unique()
    temp = go.Figure()

    t = dataFrame.sample(n = 500*len(locations))
    t.reset_index()

    for i in range(len(locations)):
        sel = t.loc[t.Location.isin([locations[i]])]
        temp = temp.add_trace(go.Scatter(y = sel["Temperature"], x = sel["Demand"],
            mode = "markers", name = locations[i]))

    return temp

if __name__ == "__main__":
    app.run_server(debug=True)