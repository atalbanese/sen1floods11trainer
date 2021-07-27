import rasterio
import numpy as np
from pathlib import Path
import pandas as pd
import math


# Return a list of file paths given two folders, imagery and labels
def get_data_list(imagery_folder, label_folder):
    file_list = []
    # Check the paths then get images and labels from folders
    if label_folder.exists() & imagery_folder.exists():
        # Convert generators to lists, ensure they are sorted the same way since generator output is arbitrary.
        images = sorted(list(imagery_folder.iterdir()))
        labels = sorted(list(label_folder.iterdir()))

        # Create a list of image,label tuples
        file_list = [(images[x], labels[x]) for x in range(0, len(images))]
    else:
        message = "Files not found. Check your folder structure"
        return print(message)
    return file_list


# Return a list of file paths for each dataset
def get_data(data_label):
    data_list = []
    if data_label == "S1Hand":
        data_list = get_data_list(Path('./files/data/flood_events/HandLabeled/S1Hand'),
                                  Path('./files/data/flood_events/HandLabeled/LabelHand'))
    elif data_label == "S2Hand":
        data_list = get_data_list(Path('./files/data/flood_events/HandLabeled/S2Hand'),
                                  Path('./files/data/flood_events/HandLabeled/LabelHand'))
    elif data_label == "S1OtsuHand":
        data_list = get_data_list(Path('./files/data/flood_events/HandLabeled/S1OtsuLabelHand'),
                                  Path('./files/data/flood_events/HandLabeled/LabelHand'))
    elif data_label == "JRCHand":
        data_list = get_data_list(Path('./files/data/flood_events/HandLabeled/JRCWaterHand'),
                                  Path('./files/data/flood_events/HandLabeled/LabelHand'))
    elif data_label == "S1Weak":
        data_list = get_data_list(Path('./files/data/flood_events/WeaklyLabeled/S1Weak'),
                                  Path('./files/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak'))
    elif data_label == "S2Weak":
        data_list = get_data_list(Path('./files/data/flood_events/WeaklyLabeled/S1Weak'),
                                  Path('./files/data/flood_events/WeaklyLabeled/S2IndexLabelWeak'))
    elif data_label == "Perm":
        data_list = get_data_list(Path('./files/data/perm_water/S1Perm'),
                                  Path('./files/data/perm_water/JRCPerm'))
    return data_list


# Create a dataframe of all files in a dataset sorted by total water pixels. This also filters out Bolivian data as a
# separate test group
def create_df(data_list):
    data_list = [item for item in data_list if item[0].parts[-1].find("Bolivia") == -1]
    df = pd.DataFrame(data_list, columns=["Image", "Label"])
    # Find total water pixels in each image
    total_water_labels = []
    for item in data_list:
        label = rasterio.open(item[1]).read().squeeze(0)
        label = label.copy()
        label[label == -1] = 0
        total_water_labels.append(np.sum(label))
    df["total_water_pix"] = total_water_labels
    df = df.sort_values(by=["total_water_pix"], ascending=False)
    df["group"] = create_groups_list(len(df), 10)
    return df


# Splits each dataset into groups of 10. Allows us to randomly sample from small groups sorted by # of water pixels
def create_groups_list(list_length, group_size):
    groups_out = []
    for i in range(0, list_length):
        groups_out.append(math.floor(i/group_size))
    return groups_out

# Split dataframe along provided porportions tuple
def prop_split(df, props):
    #get proportions
    train_prop, valid_prop, test_prop = props
    # sample for training proportion and drop selected samples
    train_df = df.groupby(['group']).sample(frac=train_prop)
    df = df.drop(train_df.index)
    # sample for valid proportion, after recalculating valid proportion to fit remaining proportion of df
    valid_prop = round(valid_prop/(1-train_prop), 1)
    valid_df = df.groupby(['group']).sample(frac=valid_prop)
    # everything thats left is the test proportion
    test_df = df.drop(valid_df.index)
    return train_df, valid_df, test_df


# For given list of dataset types, create a dataframe of that type and split it into train, test, validation data
def make_splits(types):
    for group in types:
        main_df = create_df(get_data(group))
        train, valid, test = prop_split(main_df, (0.6,0.2,0.2))
        train.to_csv(Path('./splits/' + group + '/train_split.csv'), columns=["Image", "Label"], header=False, index=False)
        valid.to_csv(Path('./splits/' + group + '/valid_split.csv'), columns=["Image", "Label"], header=False, index=False)
        test.to_csv(Path('./splits/' + group + '/test_split.csv'), columns=["Image", "Label"], header=False, index=False)


types_list = ["S1Hand", "S2Hand", "S1OtsuHand", "JRCHand", "S1Weak", "S2Weak", "Perm"]
make_splits(types_list)
