# Data load

# loading modules
import numpy as np
import pandas as pd

### LOADING THE DATA

# loading the codon usage data using the Pandas library
filename = "C:/Users/jonas/OneDrive/Skrivebord/MyUnixWorkplace/Machine Learning and Data Mining\Project\data\codon_usage.csv"
df = pd.read_csv(filename, skiprows=[487, 5064])
# skipping rows with erroneous data
# for instance, row 487 has "non-B hepatitis virus" in its "UUU" column,
# and 5064 has "12;l" and "-" in its "UUU" and "UUC" columns, respectively.
#df = df.drop(df.index[486])

# converting to numpy arrays
raw_data = df.values

# creating the X data matrix
X = raw_data[:, 5:]
# change from data type object to float
X = X.astype('float64')

# attribute extraction
attributeNames = np.asarray(df.columns[5:])

# conversion of class strings to indexes (numerical values)
# step 1: String extraction (the kingdom of each organism)
classLabels = raw_data[:, 0]
# filtering for redundancy
classNames = np.unique(classLabels)
# indexing
classDict = dict(zip(classNames, range(len(classNames))))
# creating the class index vector y
# for each class (string) in the data set (for each row),
# convert it to its index (numerical value)
y = np.array([classDict[cl] for cl in classLabels])

# defining the number of objects and the number of attributes (N & M)
N, M = X.shape

# defining the number of classes (C)
C = len(classNames)

