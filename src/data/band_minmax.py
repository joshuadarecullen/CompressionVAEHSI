import json
from tqdm import tqdm
import numpy as np

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.bigearth_datamodule import BigEarthDataModule

data_dir = "data/"

dm = BigEarthDataModule(
        dataset_dir=data_dir,
        train_val_test_split=(1.0, 0.0, 0.0),
        batch_size=32,
        bands='all',
        )
dm.setup()

band_values = {i: {'min': np.inf, 'max':-np.inf} for i in range(14)}

print(f"Amount of Examples: {len(dm.train_dataset)}")

#TODO: use val loader (no shuffling, keep track of min and max values stored to see if there are any outliers). create a df for min and max values for all examples

for data in tqdm(dm.train_dataset): #idx in range(len(data.folders)):
    image = data['image']
    for band in range(14):
        band_data = image[band]

        if band not in band_values.keys() or band_data.min().item() < band_values[band]['min']:
            band_values[band]['min'] = band_data.min().item()

        if band not in band_values.keys() or band_data.max().item() >  band_values[band]['max']:
            band_values[band]['max'] = band_data.max().item()
    break

# Convert the dictionary to a JSON string
dict_str = json.dumps(band_values, indent=4)

# Write the string to a text file
with open('band_minmax.txt', 'w') as file:
    file.write(dict_str)
