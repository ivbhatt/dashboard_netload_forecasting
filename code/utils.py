import sys, os

import numpy as np
import pandas as pd

from dash import html


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def info(func):
    def inner(*args, **kwargs):
        print(bcolors.OKCYAN + "[INFO]", end = "\t")
        func(*args, **kwargs)
        print(bcolors.ENDC, end = "")
    return inner

def warn(func):
    def inner(*args, **kwargs):
        print(bcolors.WARNING + "[WARN]", end = "\t")
        func(*args, **kwargs)
        print(bcolors.ENDC, end = "")

    return inner

DATA_PATH = os.path.join("..", "data")



def farenheit_2_celcius(series):
    return (series - 32)*5/9    


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
                        temp_df.drop(columns=["Net"], inplace = True)


                    temp_df["Dataset"] = dataset
                    temp_df["Location"] = file.split(".")[0]
                    temp_df["Year"] = year_series
                    temp_df["Month"] = month_series
                    temp_df["Day"] = date["Day"]
                    temp_df["Hour"] = date["Hour"]
                    temp_df["Weekday"] = date["Weekday"]
                    temp_df["Temperature"] = t["Temperature"]
                    
                    # print(dataset)
                    # if dataset in ["S1", "S2"]:
                    #     temp = temp_df["Demand"] / 1000 
                    #     print(temp.head())
                    #     temp_df.drop(columns=["Demand"], inplace=True) 
                    #     print(temp.head())
                    #     temp_df["Demand"] = temp[:]
                    #     print(temp.head())
                        

                    dataFrame = pd.concat([dataFrame, temp_df])

        else:
            temp_df = pd.read_csv(os.path.join(root, files[0]))[["Month","Day","Weekday","Hour","Demand", "Temperature"]]
            temp_df["Dataset"] = dataset
            temp_df["Location"] = "ALL"
            temp_df["Year"] = "ALL"


            if dataset in ["S1", "S2"]:
                temp = temp_df["Demand"] / 1000 
                temp_df.drop(columns=["Demand"], inplace=True) 
                temp_df["Demand"] = temp[:]

                temp = farenheit_2_celcius(temp_df["Temperature"])
                temp_df.drop(columns=["Temperature"], inplace=True) 
                temp_df["Temperature"] = temp[:]
            
            dataFrame = pd.concat([dataFrame, temp_df])
    return dataFrame



def load_scores(dataset):
    print_info("Working on dataset:", dataset)
    dataFrame = pd.DataFrame(columns=["Dataset", "Location", "Model", "Metric", "Value"])

    
    with open(os.path.join(DATA_PATH, dataset, "result", "Scores.csv"), "r") as scores_file:
        
        headers = scores_file.readline().split(",")
        scores = scores_file.readline().split(",")

        for (i, header) in enumerate(headers):
            location = "ALL"
            if " " in header:
                location = header.split(" ")[0]
            header = header.split(" ")[-1]

            model = header.split("_")[0]
            metric = header.split("_")[1]

            value = round(scores[i], 2)
            



            
        dataFrame = pd.concat([dataFrame, [dataset, location, model, metric, value]])
    return dataFrame



@info
def print_info(*args, **kwargs):
    print(*args, **kwargs)

@warn
def print_warn(*args, **kwargs):
    print(*args, **kwargs)

def convert_to_message(status):
    result = []
    result.append("Dataset Selected: " + str(status["dataset"]))
    result.append(html.Br())
    result.append("Locations Selected: " + ",".join(status["locations"]))
    result.append(html.Br())
    result.append("Years Selected: " + ",".join([str(i) for i in status["years"]]))
    result.append(html.Br())
    print(result)
    return result

def normalize(arr):
    zero_one = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return zero_one*2 -1

