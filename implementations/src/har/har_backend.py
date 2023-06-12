import pandas as pd
import numpy as np
import random
import os

# Concatenates all har data into a single dataframe
def aggregate_har_data(directory = '../resources/labeled_data'):
    data = []
    for file in os.listdir(directory):
        data.append(pd.read_csv(directory+'/'+file))
    return pd.concat(data)

# Bootstrapped subsequence sampling from a stepped sliding window, with a random initialization
def sample_slide_comile(df, window, step = 1, sample = 100):
    data = []
    labels = []
    for activity in df.activity.unique():
        for person in df.person.unique():
            d = df[(df.person == person)&(df.activity == activity)].reset_index(drop=True)
            x = np.array(d.X)
            y = d.activity
            temp_data = []
            temp_labels = []
            if len(x)>0:
                a = 0
                while len(temp_data)<sample:
                    rano = random.randint(0, len(x)) if a != 0 else a
                    a += 1
                    for i in range(rano, len(x) - window, step):
                        if len(temp_data)>=sample:
                            break
                        temp_labels.append(y[i])
                        temp_data.append(np.array(x[i:i+window]))
                data.extend(temp_data[:sample])
                labels.extend(temp_labels[:sample])
    return data, labels

# Simple stepped sliding window for subsequence extraction
def slide_compile(df, window, step = 1, sample = 100):
    data = []
    labels = []
    for activity in df.activity.unique():
        for person in df.person.unique():
            d = df[(df.person == person)&(df.activity == activity)].reset_index(drop=True)
            x = np.array(d.X)
            y = activity
            temp_data = []
            temp_labels = []
            for i in range(0, len(x) - window, step):
                if len(temp_data) >= sample:
                    break
                temp_labels.append(y)
                temp_data.append(np.array(x[i:i+window]))
            data.extend(temp_data[:sample])
            labels.extend(temp_labels[:sample])

    return data, labels

# Extracts randomly positioned subsequences
def random_compile(df, window, sample = 300, step = 1):
    data = []
    labels = []
    for activity in df.activity.unique():
        for person in df.person.unique():
            d = df[(df.person == person)&(df.activity == activity)].reset_index(drop=True)
            x = np.array(d.X)
            y = d.activity
            if len(x) > window:
                for n in range(sample):
                    indexer = random.randint(window//2, len(x) - window//2)
                    series = x[indexer-window//2:indexer+window//2]
                    data.append(np.array(series))
                    labels.append(y[indexer])

    return data, labels

compilers = {'random' : random_compile,
             'slide' : slide_compile,
             'sample_slide' : sample_slide_comile,
             'aggregate' : aggregate_har_data,
            }