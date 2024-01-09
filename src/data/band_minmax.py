# import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.bigearth_dataset import BigEarthDataset

data_dir = "/home/joshua/Documents/phd/CompressionVAEHSI/data"
master = "/home/joshua/Documents/phd/CompressionVAEHSI/data/master.csv"

df = pd.read_csv(master)

data = BigEarthDataset(
    root=data_dir,
    bands='all',
    download=False,
)


print(f"Amount of Examples: {len(data)}")

#TODO: use val loader (no shuffling, keep track of min and max values stored to see if there are any outliers). create a df for min and max values for all examples

df['bands_minmax'] = None

for i, item in tqdm(enumerate(data), total=len(data)): #idx in range(len(data.folders)):
    band_values = {j: {'min': np.inf, 'max':-np.inf} for j in range(14)}
    image = item['image']

    for band in range(14):
        band_data = image[band]

        if band not in band_values.keys() or band_data.min().item() < band_values[band]['min']:
            band_values[band]['min'] = band_data.min().item()

        if band not in band_values.keys() or band_data.max().item() >  band_values[band]['max']:
            band_values[band]['max'] = band_data.max().item()

    df['bands_minmax'].loc[i] = band_values


df.to_csv(master, index=False)
# # Convert the dictionary to a JSON string
# dict_str = json.dumps(band_values, indent=4)

# # Write the string to a text file
# with open('band_minmax.txt', 'w') as file:
#     file.write(dict_str)
