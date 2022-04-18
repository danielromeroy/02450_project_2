import numpy as np
import pandas as pd

# load codon usage data using with Pandas
filename = "./data/codon_usage.csv"
df = pd.read_csv(filename, skiprows=[487, 5064])  # skipping rows with erroneous data

raw_data = df.values  # converting to numpy arrays

X = raw_data[:, 5:]  # creating the X data matrix
X = X.astype('float64')  # change from data type object to float

attributeNames = np.asarray(df.columns[5:])  # attribute extraction

# Conversion of class strings to indexes (numerical values)

classLabels = raw_data[:, 0]  # String extraction (the kingdom of each organism)
classNames = np.unique(classLabels)  # filtering for redundancy
classDict = dict(zip(classNames, range(len(classNames))))  # indexing

# creating the class index vector y
# for each class (string) in the data set (for each row),
# convert it to its index (numerical value)
y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape  # number of objects and attributes

C = len(classNames)  # number of classes (C)
