import numpy as np

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





@info
def print_info(*args, **kwargs):
    print(*args, **kwargs)

@warn
def print_warn(*args, **kwargs):
    print(*args, **kwargs)

def convert_to_message(status):
    result = "Dataset Selected:" + str(status["dataset"]) + "<br>"
    result +="Locations Selected:" + ",".join(status["locations"]) + "<br>"
    result +="Years Selected:" + ",".join([str(i) for i in status["years"]]) + "<br>"
    
    return result

def normalize(arr):
    zero_one = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return zero_one*2 -1

