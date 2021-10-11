import os, sys
from dash.dcc.Checklist import Checklist
from dash.dcc.RadioItems import RadioItems
from dash.dependencies import Input, Output
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

from utils import print_info, print_warn, convert_to_message

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
            temp_df["Location"] = "ALL"
            temp_df["Year"] = "ALL"

            dataFrame = pd.concat([dataFrame, temp_df])
    return dataFrame

dataFrames = {}
for dataset in datasets:
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
month_hour_heatmap = px.imshow(np.array(dataFrame.groupby(["Month", "Hour"])["Demand"].median()).reshape(12,24), color_continuous_scale="icefire")
location_correlation = px.imshow(np.array([1]).reshape(1,1), color_continuous_scale="icefire")

## line chart
autocorrelation = go.Figure()

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
        ], value=["ALL"])
    ], style = {"border" : "1px solid black"}
    ),
    dcc.Graph(id = "hour_demand", figure=hour_demand),
    dcc.Graph(id = "month_demand", figure=month_demand),
    dcc.Graph(id = "year_demand", figure=year_demand),
    dcc.Graph(id = "month_hour_heatmap",figure = month_hour_heatmap),
    dcc.Graph(id = "location_correlation",figure = location_correlation),
    dcc.Graph(id = "autocorrelation", figure = autocorrelation)

])


@app.callback(
    output = {"location_selector_options":Output(component_id="location_selector", component_property="options"),
            "location_selector_value":Output(component_id="location_selector", component_property="value"),
            "year_selector_options":Output(component_id="year_selector", component_property="options"),
            "year_selector_value":Output(component_id="year_selector", component_property="value"),
            "active_dataset":Output(component_id="active_dataset", component_property="children")
    },
    inputs ={
        "selected_dataset":Input(component_id="data_selector", component_property="value"),
        "selected_locations":Input(component_id="location_selector", component_property="value"),
        "selected_years":Input(component_id="year_selector", component_property="value")
    })

def load_new_datasset(selected_dataset, selected_locations, selected_years):
    global dataFrame
    global current_selection

    # update form fields
    # update global current_selection
    # update dataFrame
    # update active_dataset hence triggerring graph updates

    if current_selection["dataset"] != selected_dataset:
    # if True:
      
        current_selection["dataset"] = selected_dataset
        current_selection["locations"] = ["ALL"]
        current_selection["years"] = ["ALL"]
        

        dataFrame = dataFrames[selected_dataset]

        returnLocations = [{"label":"ALL", "value":"ALL"}]
        for location in dataFrame.Location.unique():
            if location != "ALL":
                returnLocations.append({"label":location, "value":location})
        returnLocation = ["ALL"]

        returnYears = [{"label":"ALL", "value":"ALL"}]
        for year in dataFrame.Year.unique():
            if year != "ALL":
                returnYears.append({"label":year, "value":year})
        returnYear = ["ALL"]

        return {
            "location_selector_options": returnLocations,
            "location_selector_value": returnLocation,
            "year_selector_options": returnYears,
            "year_selector_value": returnYear,
            "active_dataset": convert_to_message(current_selection)
        }
    else:
        returnLocations = [{"label":"ALL", "value":"ALL"}]
        for location in dataFrame.Location.unique():
            if location != "ALL":
                returnLocations.append({"label":location, "value":location})

        returnYears = [{"label":"ALL", "value":"ALL"}]
        for year in dataFrame.Year.unique():
            if year != "ALL":
                returnYears.append({"label":year, "value":year})

        current_selection["locations"] = selected_locations
        current_selection["years"] = selected_years

        if len(selected_locations) > 1 and "ALL" in selected_locations:
            current_selection["locations"].remove("ALL")             
        if len(selected_years) > 1 and "ALL" in selected_years:
            current_selection["years"].remove("ALL")
        

        print(current_selection)
        print(dataFrame.head())
        dataFrame = dataFrames[current_selection["dataset"]]
        if "ALL" not in current_selection["years"]:
            dataFrame = dataFrame.loc[dataFrame.Year.isin(current_selection["years"])]
        if "ALL" not in current_selection["locations"]:
            dataFrame = dataFrame.loc[dataFrame.Location.isin(current_selection["locations"])]
        print(dataFrame.shape)

        
        
        return {
            "location_selector_options": returnLocations,
            "location_selector_value": current_selection["locations"],
            "year_selector_options": returnYears,
            "year_selector_value": current_selection["years"],
            "active_dataset": convert_to_message(current_selection)
        }

                    


    



    # dataFrame = dataFrames[selected_dataset]

    # current_selection["dataset"] = selected_dataset
    # current_selection["locations"] = dataFrame.Location.unique()
    # current_selection["years"] = dataFrame.Year.unique()



    # print(current_selection)

    



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
    month_hour_heatmap = px.imshow(np.array(dataFrame.groupby(["Month", "Hour"])["Demand"].median()).reshape(12,24), color_continuous_scale="icefire")
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
            
            corrs[i][j] = np.corrcoef(a, b)[0, 1]
            corrs[j][i] = corrs[i][j]

    location_correlation = px.imshow(corrs)
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
        # a = a[:8736]
        # corrs[i] = np.correlate(a, a, mode = "full")[len(a)//2:len(a)//2 + 24]
        corrs[i] = np.correlate(a, a, mode = "full")[-24*8:]
        
        corrs[i] /= max(corrs[i])
        
        
        temp = temp.add_trace(go.Scatter(y = corrs[i], x = [i for i in range(24*8)],
            mode = "lines+markers", name = locations[i]))

    return temp

if __name__ == "__main__":
    app.run_server(debug=True)