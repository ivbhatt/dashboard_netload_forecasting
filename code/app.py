# utility libraries
import os, sys

# importing dash
import dash
from dash import html, dcc

# importing plotly components
import plotly.graph_objects as go
import plotly.express as px

# Dash components
from dash.dcc.Checklist import Checklist
from dash.dcc.RadioItems import RadioItems
from dash.dependencies import Input, Output, State
from dash.html.Label import Label

# Math and data handling libraries
import numpy as np
import pandas as pd

# custom functions
from utils import print_info, print_warn, convert_to_message
from utils import normalize, load_dataset, DATA_PATH

# production server
from waitress import serve


## constants
BOOTSTRAP_JS = "https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"   ## Bootstrap5 JS
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"       ## Bootstrap5 CSS
CUSTOM_CSS = "styles.css"                                                                        ## Custom CSS

## global variable to keep track of the current data selection
current_selection = {
    "dataset" : "S1",
    "locations" : ["ALL"],
    "years" : ["ALL"]
}

## Load the dataset information in RAM
datasets = os.listdir(DATA_PATH)
if len(datasets) != 8:
    print_warn("All 8 datasets not found." )
print_info("Datasets:", datasets)


# We now load the data into RAM
### FOR DEBUG #########################################
#  S1 and S4 are two good datasets that load fast
#  Use only S1 and S4 when trying to add a new feature

dataFrames = {}
# for dataset in ["S1", "S4"]:
#     dataFrames[dataset] = load_dataset(dataset)
for dataset in datasets:
    dataFrames[dataset] = load_dataset(dataset)
#######################################################

# Select dataset S1 by default
dataFrame = dataFrames["S1"]

## HOUR BREAKUP ############################
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
############################################

## MONTH BREAKUP ###########################
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
############################################

## YEAR BREAKUP ############################
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
############################################

# heatmaps #################################
month_hour_heatmap_data = np.array(dataFrame.groupby(["Month", "Hour"])["Demand"].median()).reshape(12,24)
month_hour_heatmap = px.imshow(normalize(month_hour_heatmap_data), color_continuous_scale="Bluered")

location_correlation = px.imshow(np.array([1]).reshape(1,1), color_continuous_scale="Bluered")
location_correlation.layout.height = 500
############################################

## line chart ##############################
autocorrelation = go.Figure()
############################################

## scatter plot ############################
temp_demand_correlation = go.Figure()
############################################


# creating the Dash app instance
app = dash.Dash(__name__, external_stylesheets=[BOOTSTRAP_CSS,], external_scripts=[BOOTSTRAP_JS])

## This is where the "html" layout is written
app.layout = html.Div(id = "main-block", className="container", children=[
    html.Div(id = "heading-block", className = "container", children = [
        html.H1(id = "heading", children="Solar power dashboard")
    ]),
    
    html.Div(id = "status-block", className = "container", children = [
        html.P(className = "alert alert-warning", children = [
            "S3 and L3 are large datasets and it takes more time to load them."
        ]),
        html.P(id ="active_dataset", children=
            convert_to_message(current_selection)
        ),
    ]),

    html.Div(id = "selector", className = "container",
    style = {"border" : "1px solid black"}, children = [
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
        html.Label("Select the Location:"),
        dcc.Checklist( id = "location_selector", options = [
            {"label": "ALL", "value":"ALL"}
        ], value=["ALL"]),
        html.Label("Select the Years:"),
        dcc.Checklist( id = "year_selector", options = [
            {"label": "ALL", "value":"ALL"}
        ], value=["ALL"]),
        html.Button(id = "submit", type = "submit", 
        className = "btn btn-primary", children = "Refresh!")
    ]),
    html.Div(id = "hour_demand-block", className = "container", children = [
        dcc.Graph(id = "hour_demand", figure=hour_demand)
    ]),
    html.Div(id = "month_demand-block", className = "container", children = [
        dcc.Graph(id = "month_demand", figure=month_demand)
    ]),
    html.Div(id = "year_demand-block", className = "container", children = [
        dcc.Graph(id = "year_demand", figure=year_demand)
    ]),
    html.Div(id = "month_hour_heatmap-block", className = "container", children = [
        dcc.Graph(id = "month_hour_heatmap",figure = month_hour_heatmap)
    ]),
    html.Div(id = "location_correlation-block", className = "container", children = [
        dcc.Graph(id = "location_correlation",figure = location_correlation)
    ]),
    html.Div(id = "autocorrelation-block", className = "container", children = [
        dcc.Graph(id = "autocorrelation", figure = autocorrelation)
    ]),
    html.Div(id = "temp_demand_correlation-block", className = "container", children = [
        dcc.Graph(id = "temp_demand_correlation", figure = temp_demand_correlation)
    ]),
    html.Div(id = "footer-block", children = [
        html.P(children = [
            html.Br(),
            "Built with dash-plotly",
            html.Br(),
            "Built at NC State University, Raleigh"
        ])
    ])
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
    global dataFrame
    global current_selection

    ## This call-back does 4 things:
    # update form fields
    # update global current_selection
    # update dataFrame
    # update active_dataset hence triggerring graph updates

    if current_selection["dataset"] != selected_dataset:

        # update dataFrame    
        dataFrame = dataFrames[selected_dataset]

        # update global current_selection
        current_selection["dataset"] = selected_dataset
        current_selection["locations"] = [i for i in dataFrame.Location.unique()]
        current_selection["years"] = [i for i in dataFrame.Year.unique()]

        # update form fields
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

        # update form fields
    
        returnLocations = []
        for location in dataFrames[current_selection["dataset"]].Location.unique():
            returnLocations.append({"label":location, "value":location})

        returnYears = []
        for year in dataFrames[current_selection["dataset"]].Year.unique():
            returnYears.append({"label":year, "value":year})


        # update global current_selection
        current_selection["locations"] = selected_locations
        current_selection["years"] = selected_years


        # update dataFrame
        dataFrame = dataFrames[current_selection["dataset"]]    
        dataFrame = dataFrame.loc[dataFrame.Year.isin(current_selection["years"])]
        dataFrame = dataFrame.loc[dataFrame.Location.isin(current_selection["locations"])]
        
        
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
def update_hour_demand(selected_dataset):
    # Update this graph according to the global variable dataFrame
    # this should be up-to date!

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

    # Adding labels and title
    hour_demand.update_layout(
        boxmode="group",
        title="Demand vs Time-of-Day",
        xaxis_title="Hour of Day",
        yaxis_title = "Demand (MW)"
    )

    

    return hour_demand

@app.callback(
    Output(component_id="month_demand", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_month_demand(selected_dataset):

    # Update this graph according to the global variable dataFrame
    # this should be up-to date!

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

    # Adding labels and title
    month_demand.update_layout(
        boxmode="group",
        title="Demand vs Time-of-Year",
        xaxis_title="Month",
        xaxis = {
            "tickmode" : "array",
            "tickvals" : [i for i in range(12)],
            "ticktext" : ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        },
        yaxis_title = "Demand (MW)"
        )
    return month_demand

@app.callback(
    Output(component_id="year_demand", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_year_demand(selected_dataset):

    # Update this graph according to the global variable dataFrame
    # this should be up-to date!
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

    # Adding labels and title
    year_demand.update_layout(
        boxmode="group",
        title = "Demand over the years",
        xaxis_title="Year",
        yaxis_title = "Demand (MW)"
        )
    return year_demand

@app.callback(
    Output(component_id="month_hour_heatmap", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_month_hour_heatmap(selected_dataset):
    
    # Update this graph according to the global variable dataFrame
    # this should be up-to date!    
    month_hour_heatmap_data = np.array(dataFrame.groupby(["Month", "Hour"])["Demand"].median()).reshape(12,24)
    
    # Adding labels and title
    month_hour_heatmap = px.imshow(month_hour_heatmap_data, color_continuous_scale="Bluered", 
    labels = {"color": "Demand (MW)"})
    
    month_hour_heatmap.update_layout(
        xaxis_title="Time-of-Day",
        yaxis = {
            "tickmode" : "array",
            "tickvals" : [i for i in range(12)],
            "ticktext" : ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        },
        yaxis_title = "Month",
        title = "Demand Month-Hour Signature"

    )
    return month_hour_heatmap

@app.callback(
    Output(component_id="location_correlation", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_location_correlation(selected_dataset):

    # Update this graph according to the global variable dataFrame
    # this should be up-to date!    
    locations = dataFrame.Location.unique()

    corrs = [[ 0 for i in range(len(locations)) ] for k in range(len(locations))]

    for i in range(len(locations)):
        for j in range(i, len(locations)):

            a = dataFrame.loc[dataFrame.Location.isin([locations[i]])].sort_values(by = ["Year", "Month", "Day", "Hour"])["Demand"]
            b = dataFrame.loc[dataFrame.Location.isin([locations[j]])].sort_values(by = ["Year", "Month", "Day", "Hour"])["Demand"]
            
            # np.corrcoef give the pearson's correlation coeffiecient between two attributes
            t = np.corrcoef(a, b)
            
            # In case there is only one location, we need this kind of checker
            if len(t) == 1:
                corrs[i][j] = t
            else:
                corrs[i][j] = t[0][1]

            corrs[j][i] = corrs[i][j]

    location_correlation = px.imshow(corrs, color_continuous_scale="Bluered",
    
    # Adding labels and title
    labels = {"x" : "Location", "y": "Location", "color":"Correlation-Coefficient"},
    x = current_selection["locations"],
    y = current_selection["locations"],
    )

    location_correlation.update_layout(title = "Correlation between Locations")    

    return location_correlation


@app.callback(
    Output(component_id="autocorrelation", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_autocorrelation(selected_dataset):

    # Update this graph according to the global variable dataFrame
    # this should be up-to date!    

    locations = dataFrame.Location.unique()
    temp = go.Figure()

    
    for i in range(len(locations)):
        a = dataFrame.loc[dataFrame.Location.isin([locations[i]])].sort_values(by = ["Year", "Month", "Day", "Hour"])
        a = a["Demand"]

        # We use normalized autocorrelation ###################
        ## More information at: https://stackoverflow.com/a/676302
        t = np.correlate(a, a, mode = "full")
        corrs = t[len(t)//2: 201 + len(t)//2][:]
        corrs /= max(corrs)
        ###########################################################

        temp = temp.add_trace(go.Scatter(y = corrs, x = [i for i in range(201)],
            mode = "lines+markers", name = locations[i]))
    
    # Adding labels and title
    temp.update_layout(
        title = "Autocorrelation on Demand",
        xaxis_title="Time-lag (Hours)",
        yaxis_title = "Correlation-coefficient"
    )

    return temp


@app.callback(
    Output(component_id="temp_demand_correlation", component_property="figure"),
    Input(component_id="active_dataset", component_property="children"))
def update_temp_demand_correlation(selected_dataset):
    locations = dataFrame.Location.unique()
    temp = go.Figure()

    t = dataFrame.sample(n = 500*len(locations))

    t["Temperature"] = np.array(t["Temperature"])
    t["Demand"] = np.array(t["Demand"])

    t.reset_index()

    for i in range(len(locations)):
        sel = t.loc[t.Location.isin([locations[i]])]
        temp = temp.add_trace(go.Scatter(y = sel["Temperature"], x = sel["Demand"],
            mode = "markers", name = locations[i]))

    # Adding labels and title
    temp.update_layout(
        title = "Demand vs Temperature",
        xaxis_title="Demand (MW)",
        yaxis_title = "Temperature (Celsius-scale)"
    )
    return temp

if __name__ == "__main__":
    serve(app.server, host="0.0.0.0", port=8050)
