import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import math

def check_input(data):
    if (data.shape[1] == 2):
        if (isinstance(data, pd.DataFrame)) & (is_string_dtype(data.iloc[:,0])) & (is_numeric_dtype(data.iloc[:,1])):
            return True
        else:
            return False
    else:
        return False

def create_all_coalitions(channels):
    from itertools import combinations
    
    all_coalitions = []
    repeat = len(channels)
    
    for i in range(1,repeat+1):
        for s in combinations(channels, i):
            all_coalitions.append(tuple(sorted(s)))
        
    return all_coalitions


def find_NAs(characteristic_function):
    data = characteristic_function
    data["metric"].fillna(0, inplace = True)
    for i in data[data["channels"].isnull()].index.values:
        cell = data["coalitions"][i]
        if len(cell) == 1:
            data.loc[i, "channels"] = str(cell).replace("(","").replace(")","").replace(",","").replace("'","")
        else:
            data.loc[i, "channels"] = str(cell).replace("(","").replace(")","").replace("'","").replace(" ","")
        data.loc[i,"length"] = len(data["channels"][i].split(","))
    return data    


def find_all_channels(data):
    all_channels = []
    for channel in data["channels"]:
        for c in channel.split(','):
            all_channels.append(c)

    return list(dict.fromkeys(all_channels))


def add_length(data):
    length = []
    
    for i, c in enumerate(data["channels"]):
        length.append(len(c.split(",")))

    data["length"] = length
    
    return data


def order_channels(data):
    d = []
    for c in data:
        temp = c.split(",")
        temp.sort()
        d.append(','.join(temp))
    return d


def marginal_value(channel, channels, characteristic_function): 
    # create a local copy of the characteristic function
    char_func = characteristic_function.copy()   
    # Order the characteristic function by length descending
    char_func.sort_values(by = "length", ascending = False, inplace = True)
    char_func.reset_index(inplace = True)
    
    values = []
    
    L = len(channels[0].split(","))
    # We are only interested in the coalitions where channel was a part of
    coalitions = char_func[(char_func["channels"].str.contains(channel)) &
                          (char_func["length"] <= L)]["channels"]
    # For each channel combinations in the characteristic function
    for c in coalitions:
        l = len(c.split(","))
            
        # Remove cases where for example where channels == 'a,c' but 'a,b' popped up in the coalitions list because
        # the selected channel is 'a'. We want to make sure that we do not include 'a,b' related calcs
        if (l == L) and c != channels[0]:
            continue
            
        else:
            if l == 1:               
                try:
                    factor = math.factorial((l-1)) * math.factorial(L - (l - 1) - 1) / math.factorial(L)
                    marginal_value = char_func[(char_func["channels"].str.contains(c)) &
                                  (char_func["length"] == l)]["metric"].item()
                    values.append(marginal_value * factor)
                except ValueError:
                    continue
            else:
                try:
                    v1 = char_func[(char_func["channels"].str.contains(c)) &
                                  (char_func["length"] == l)]["metric"].item()                    
            
                    coalition_without_channel = c.split(",")
                    coalition_without_channel.remove(channel)
                    coalition_without_channel = ",".join(coalition_without_channel)

                    v2 = char_func[(char_func["channels"].str.contains(coalition_without_channel)) &
                                  (char_func["length"] == l - 1)]["metric"].item()
                    marginal_value = v1 - v2
                    factor = math.factorial((l-1)) * math.factorial(L - (l - 1) - 1) / math.factorial(L)
                    values.append(marginal_value*factor)
                    
                except ValueError:
                    continue
                    
    if len(values) == 0:
        return -1
    else:
        return np.sum(values)
