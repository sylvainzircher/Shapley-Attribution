from .functions import create_all_coalitions
from .functions import find_NAs
from .functions import find_all_channels
from .functions import add_length
from .functions import order_channels
from .functions import marginal_value
import pandas as pd
import numpy as np
import math

def shapley(original_dataset):
    # Make a copy of the original dataset not to change it directly
    data = original_dataset.copy()
    data.columns = ["channels","metric"]
    # Add column length to final dataset
    data = add_length(data)
    # Order the channels
    data["channels"] = order_channels(data["channels"])
    # Create tuples for joining against coalitions
    tupes = []
    for i in np.arange(len(data["channels"])):
        if len(data["channels"][i].split(",")) == 1:
            tupes.append(tuple([data["channels"][i]]))
        else:
            t = []
            for c in data["channels"][i].split(","):
                t.append(c)
            tupes.append(tuple(t))
    data["join"] = tupes
    
    # Sort the data frame by length ascending
    data.sort_values(by = ["length"], inplace = True)   
    
    # Find all uniques channels within the data
    channels = find_all_channels(data)
    # Find all coalitions for all the channels and save it in a Dataframe
    coalitions = pd.DataFrame({'coalitions': create_all_coalitions(channels)})
    # Create the characteristic function from the data we have
    coalitions_from_data = pd.DataFrame({"channels":data["channels"],
                                         "coalitions":data["join"],
                                         "metric":data["metric"],
                                         "length":data["length"]})
    # Create now the full characteristic function with the value from the data we have adding zeros for missing coalitions
    characteristic_function = pd.merge(coalitions, coalitions_from_data, how = 'left', on = "coalitions")
    characteristic_function = find_NAs(characteristic_function)

    values = []
    # We loop through any coalition (2^N - 1) combinations if we exclude the 0,0,..,0 case
    # So basically we go through the df row by row
    for i in np.arange(0, 2**len(channels) - 1):
        local_value = []
        # Find all the channels within that coalition
        local_channels = characteristic_function["channels"][i].split(",")

        # Loop through all the channels
        for c in local_channels:
            val = marginal_value(c, [",".join(local_channels)], characteristic_function)
            local_value.append(val)
        
        values.append(local_value)

    return pd.DataFrame({"Channels":characteristic_function["channels"],"Shapley Values":values}) 
