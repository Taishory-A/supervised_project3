# Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
#----------------------------

# Load dataset
df = pd.read_csv(r"file_location\breast_cancer-1.csv")

# Display dimensions of dataframe
print(df.shape)

print("-----------------------------------------------------------------------")
print(df.info())

print("-----------------------------------------------------------------------")
# Display 10 random samples of items
print(df.sample(10))

# Remove the last column .i.e the 33rd one as it is not needed + 0

df.drop(df.columns[[0, 32]], axis=1, inplace=True)
print(df.info())

print("-----------------------------------------------------------------------")
# Display statistics for numeric columns
print(df.describe())

# Lets know how many values for malignant and for benign type of cancer
print ("Total number of diagnosis are ", df.shape[0], ", ", 
       df.diagnosis.value_counts()["B"], "Benign and Malignant are",
       df.diagnosis.value_counts()["M"])
print("\n")
print("diagnosis        : \n",df["diagnosis"].value_counts())
print("\n")
print(df["diagnosis"].value_counts(normalize=True))

 
df["diagnosis"].value_counts().plot.bar(title="diagnosis")
# Dropping the duplicate rows
duplicate_rows = df[df.duplicated()]
print("Number of duplicate rows : ",duplicate_rows.shape)
df.drop_duplicates(inplace=True)

# To check missing values
print("SUM : \n",df.isnull().sum())
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")


# Mapping of data feature values
# df_copy
df_copy = df.copy()
df_copy["diagnosis"] = df_copy["diagnosis"].map({"B":0,"M":1})
print(df_copy["diagnosis"].sample(10))

# Handling Outlier
fig, ax = plt.subplots(figsize=(50,16)) 
sns.boxplot(data=df_copy, orient="v", palette="Set2")
plt.show()


# One column
sns.boxplot(x= df_copy["perimeter_mean"], data= df_copy)
sns.swarmplot(x= df_copy["perimeter_mean"],  data= df_copy, color=".25")

# Remove Outlier
def remove_outlier(col):
    sorted(col)
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    print("Q1 = ", Q1, " Q3 = ", Q3, " IQR = ", IQR)
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range

# Extract all column names as list in Pandas Dataframe
lst_all_cols = df_copy.columns.tolist()
print("lst_all_cols : \n",lst_all_cols)

print("\n")
# Numerical data distribution
print(list(set(df_copy.dtypes.tolist())))
df_num = df_copy.select_dtypes(include = ["float64"])
lst_num_cols = df_num.columns.tolist()
print("Numerical Data : \n",lst_num_cols)
print("\n")


indx = 0
for col in lst_num_cols:
    print(indx)
    lower_range, upper_range =  remove_outlier(df_copy[col])
    df_copy[col] = np.where(df_copy[col] < lower_range, lower_range, df_copy[col]) 
    df_copy[col] = np.where(df_copy[col] > upper_range, upper_range, df_copy[col])
    indx = indx +1
    print("-----------------------------")


fig, ax = plt.subplots(figsize=(50,16)) 
sns.boxplot(data=df_copy, orient="v", palette="Set2")
plt.show()


# Correlation matrix 
corrMatrix = df_copy.corr()

# plot corrMatrix
fig, ax = plt.subplots(figsize=(25,20)) 
# sns.heatmap(corrMatrix, annot=True)
sns.heatmap(corrMatrix, annot=True, linewidth=0.01, square=True, cmap="RdBu", linecolor="black")


# Correlation with output variable
cor_target = abs(corrMatrix["diagnosis"])

# Selecting highly correlated features : 0.5
relevant_features = cor_target[cor_target>0.5]
print("relevant_features : ",relevant_features.shape,"\n",relevant_features)

print("-----------------------------------------------------------------------")
lst_columns = relevant_features.index.to_list()

my_data = pd.DataFrame(df_copy, columns= lst_columns)
print (my_data.sample(10))

print("-----------------------------------------------------------------------")
my_cols = my_data.columns.to_list()
print("List the column names : ",len(my_cols),"\n",my_cols)


# Define X variables and our target(y)

X = my_data.drop(["diagnosis"],axis=1).values
y = my_data["diagnosis"].values
print("X : ",X.shape," y : ",y.shape)

# Standardization

st_scaler = StandardScaler()
st_scaler.fit(X)
X_st_scaler = st_scaler.transform(X)


# Split into Input and Output Elements

X_train, X_test, y_train, y_test = train_test_split(X_st_scaler, 
              y, test_size= 0.20, random_state=100, stratify=y)

print("X_train = ",X_train.shape ," y_train = ", y_train.shape)
print("X_test  = ",X_test.shape ," y_test = ", y_test.shape)


# Create a classifier: a support vector classifier
classifier = svm.SVC(kernel="linear")


# Train the classifier
classifier.fit(X_train, y_train)

predicted = classifier.predict(X_test)

# Classification report
print("Classification report : \n", classifier,"\n", 
      metrics.classification_report(y_test, predicted))
