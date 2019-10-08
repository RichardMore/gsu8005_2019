---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python [conda env:py27]
    language: python
    name: conda-env-py27-py
---

# Crime Analysis for Atlanta


## By: Anuja Jain, Kelly Tran, Qingyuan Jiang, Richard More


### Data from: http://www.atlantapd.org/i-want-to/crime-data-downloads
### NPU map from: https://www.atlantaga.gov/government/departments/city-planning/office-of-zoning-development/neighborhood-planning-unit-npu


# Importing modules

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
from sklearn.tree import export_graphviz
from IPython.display import SVG
import matplotlib.pyplot as plt
from keplergl import KeplerGl
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np
import graphviz
import pickle
```

# Importing Atlanta data

```python
atlpd_old = pd.io.parsers.read_csv("ATLPD_2009-2018.csv")
atlpd_new = pd.io.parsers.read_csv("ATLPD_09-23-2019.csv")
```

***


# Data cleaning and transformation


## Atlanta data renaming column with typo

```python
atlpd_old["Shift Occurrence"] = atlpd_old["Shift Occurence"]
del atlpd_old["Shift Occurence"]
```

## Merging DataFrames

```python
atlpd = pd.concat([atlpd_old, atlpd_new], sort = False, ignore_index = True)
atlpd.head(5)
```

## Atlanta data cleaning 2

```python
atlpd.drop(["Report Date", "Possible Date", "Possible Time", "Apartment Office Prefix", "Apartment Number", "Location Type", "IBR Code"], axis = 1, inplace = True)
atlpd.dropna(subset = ["Neighborhood"], inplace = True)
atlpd.dropna(subset = ["NPU"], inplace = True)
atlpd["UCR Literal"] = atlpd["UCR Literal"].str.split("-", expand = True)[0]
```

## Converting Occur Date to Date type

```python
atlpd["Occur Date"] = pd.to_datetime(atlpd["Occur Date"])
```

## Dropping lines where Report Time is wrong

```python
index_wrong_time = atlpd[pd.to_numeric(atlpd["Occur Time"], errors = 'coerce').isnull()].index
atlpd.drop(index_wrong_time , inplace = True)
```

## Converting Occur Time to numeric type

```python
atlpd["Occur Time"] = pd.to_numeric(atlpd["Occur Time"], errors = 'coerce')
```

## Converting times
* Where time is >= 2400 the date has to shift to the next day
* Where time is > 2400 the time is scaled down to the 0-2400 range

```python
atlpd.loc[atlpd["Occur Time"] >= 2400, "Occur Date"] += pd.DateOffset(days = 1)
atlpd.loc[atlpd["Occur Time"] > 2400, "Occur Time"] -= 2400
```

## Convert some columns to Category type
* UCR Literal
* NPU
* Neighborhood

```python
atlpd["UCR Literal"] = pd.Categorical(atlpd["UCR Literal"])
atlpd["NPU"] = pd.Categorical(atlpd["NPU"])
atlpd["Neighborhood"] = pd.Categorical(atlpd["Neighborhood"])
```

## Additional columns
* Year from Occur Date
* Month from Occur Date
* Day of Week from Occur Date
* Day of Week number
* Hour from Occur Time
* Lethalness: crimes in the following categories are considered lethal: AGG ASSAULT, LARCENY, HOMICIDE, MANSLAUGHTER
 * Lethalness from UCR Literal
 * Lethalness Num from UCR Literal
* UCR Code from UCR Literal
* Neighborhood Code from Neighborhood

```python
atlpd["Year"] = atlpd["Occur Date"].dt.year
atlpd["Month"] = atlpd["Occur Date"].dt.month
atlpd["Day of Week"] = pd.Categorical(atlpd["Occur Date"].dt.day_name())
atlpd["DoW"] = atlpd["Occur Date"].dt.dayofweek
atlpd["Hour"] = atlpd["Occur Time"].round(-2)
atlpd["NPU Num"] = atlpd["NPU"].cat.codes
atlpd["Lethalness"] = pd.Categorical(np.where(atlpd["UCR Literal"].isin(["AGG ASSAULT", "LARCENY", "HOMICIDE", "MANSLAUGHTER"]),
                               "Lethal",
                               "Non-Lethal"))
atlpd["Lethalness Num"] = atlpd["Lethalness"].cat.codes
atlpd["UCR Code"] = atlpd["UCR Literal"].cat.codes
atlpd["Neighborhood Code"] = atlpd["Neighborhood"].cat.codes
```

## Additional time change
* After rounding the Occur Time we have 0 and 2400 both representing midnight -> change 2400 to 0

```python
atlpd.loc[atlpd["Hour"] == 2400, "Hour"] = 0
```

## Dropping lines where Occur Date is earlier than 2009 or later than 08/31/2019

```python
index_early_or_later = atlpd[(atlpd["Occur Date"].dt.year < 2009) | (atlpd["Occur Date"] >= pd.Timestamp("2019-09-01"))].index
atlpd.drop(index_early_or_later , inplace=True)
```

***


# Data Visulization and Exploratory Data Analytics


## Types of crime and numbers of reports

```python
ucr_counts = atlpd.groupby("UCR Literal")["Report Number"].count()
plt.figure(figsize = (10, 5))
plt.bar(range(len(ucr_counts)), ucr_counts, align = "center")
plt.xticks(range(len(ucr_counts)), ucr_counts.index)
plt.show()
```

## Day of week and number of reports

```python
ucr_counts = atlpd.groupby("Day of Week")["Report Number"].count()
ucr_counts = ucr_counts.reindex(['Monday', 'Tuesday', "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.figure(figsize = (10, 5))
plt.bar(range(len(ucr_counts)), ucr_counts, align = "center")
plt.xticks(range(len(ucr_counts)), ucr_counts.index)
plt.show()
```

## Different crime types by day of week

```python
for ucr in atlpd["UCR Literal"].unique():
    ucr_counts = atlpd[(atlpd["UCR Literal"] == ucr)].groupby("Day of Week")["Report Number"].count()
    ucr_counts = ucr_counts.reindex(['Monday', 'Tuesday', "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    plt.figure(figsize = (10, 5))
    plt.bar(range(len(ucr_counts)), ucr_counts, align = "center")
    plt.xticks(range(len(ucr_counts)), ucr_counts.index)
    plt.title("Number of " + ucr + " by days")
    plt.show()
```

## Number of crimes per year

```python
atlpd_prev = atlpd[(atlpd["Occur Date"].dt.year < 2019)]
year_counts = atlpd_prev.groupby("Year")["Report Number"].count()
plt.figure(figsize = (10, 5))
plt.bar(range(len(year_counts)), year_counts, align = "center")
plt.xticks(range(len(year_counts)), year_counts.index)
plt.show()
```

## Number of crimes by month

```python
month_counts = atlpd.groupby("Month")["Report Number"].count()
plt.figure(figsize = (10, 5))
plt.bar(range(len(month_counts)), month_counts, align = "center")
plt.xticks(range(len(month_counts)), month_counts.index)
plt.show()
```

## Number of crimes by the hour

```python
hour_counts = atlpd.groupby("Hour")["Report Number"].count()
plt.figure(figsize = (13, 5))
plt.bar(range(len(hour_counts)), hour_counts, align = "center")
plt.xticks(range(len(hour_counts)), hour_counts.index)
plt.show()
```

# Observations:
* The times when crimes are most likely to happen:
 * Midnight
 * Morning on the way to work
 * Lunchtime
 * Going home after work
 * When going out in the evening

```python
sns.countplot(x="Lethalness", data=atlpd)
```

## Lethalness by Area

```python
plt.figure(figsize = (16, 8))
sns.countplot(x = "NPU", hue = "Lethalness", data = atlpd)
```

### Lethalness of the crimes are higher in Area B, E and M. 
    Buckhead (B)
    Midtown (E)
    Downtown (M)


## Number of different crimes by time

```python
plt.figure(figsize = (13, 5))
for n, c in zip( atlpd['UCR Literal'].unique() , ['b','pink','y','g','r'] ):  
    x = atlpd.loc[ (atlpd["UCR Literal"] == n)].groupby(['Hour']).size().index
    y = atlpd.loc[ (atlpd["UCR Literal"] == n)].groupby(['Hour']).size()
    plt.plot(x, y, ':')
plt.annotate(xy = [1240,11861], s = '12:00 ~ 13:00')
plt.annotate(xy = [1870,12501], s = '18:00 ~ 19:00')
plt.legend(atlpd["UCR Literal"].cat.categories, loc = 2)
plt.xticks(x)
plt.show()
```

### Larceny and burglary happens mostly during the times when people are not at home


### Different type of Crime at different time
* Most Larceny happens between 12:00 ~ 21:00
* Most Burglay happens between 7:00 ~ 19:00
* Most Auto theft happens after 18:00


```python
for i in atlpd["NPU"].unique():
    plt.figure(figsize = (13, 5))
    plt.title("Crimes in NPU area " + str(i) + ", count: " + str(len(atlpd[atlpd.NPU == i]['Longitude'])) + ":")
    for n, c in zip( atlpd['UCR Literal'].unique() , ['y','r','b','g','pink'] ):
        x = atlpd.loc[ (atlpd["UCR Literal"] == n ) & (atlpd['NPU'] == i)].groupby(['Hour']).size().index
        y = atlpd.loc[ (atlpd["UCR Literal"] == n ) & (atlpd['NPU'] == i)].groupby(['Hour']).size()
        plt.plot(x, y, '-')
    plt.legend(atlpd["UCR Literal"].cat.categories, loc = 2)
    plt.xticks(x)
    plt.show()
```

***


# Tables


## Crime counts by NPU

```python
count_npu = atlpd.groupby("NPU")["Report Number"].count().rename("Crime Count", inplace = True)
npu_crimecount = pd.DataFrame(count_npu)
for ucr in atlpd["UCR Literal"].unique():
    npu_crimecount[ucr] = atlpd[(atlpd["UCR Literal"] == ucr)].groupby("NPU")["Report Number"].count()
npu_crimecount
```

### Most dangerous place to live: 
* B,E,M (Crime coount>20,000)

### Best place to live:
* A,C (Crime count<4000)


## Crime proportions by NPU

```python
npu_crimeprop = npu_crimecount.copy()
del npu_crimeprop["Crime Count"]
for ucr in atlpd["UCR Literal"].unique():
    npu_crimeprop[ucr] = npu_crimecount[ucr].div(npu_crimecount["Crime Count"], axis = 0)
npu_crimeprop = npu_crimeprop.style.format("{:.2%}")
npu_crimeprop
```

***


# Map of Atlanta with frequencies of reports

```python
with open("atl_map_conf.pickle") as f:
    atl_map_conf = pickle.load(f)
map_atl = KeplerGl(height = 800, config = atl_map_conf)
map_atl.add_data(data = atlpd, name = "ATLPD 2015-2019 Sept")
map_atl
```

## Saving map configurations

```python
with open("atl_map_conf.pickle", 'wb') as f:
    pickle.dump(map_atl.config, f)
```

***


# Predictive analysis


## K-NN

```python
feature_names = ["Hour", "DoW", "NPU Num"]
x_atlpd = atlpd[feature_names]
y_atlpd = atlpd["Lethalness Num"]
target_names = atlpd["Lethalness"].cat.categories

x_train, x_test, y_train, y_test = train_test_split(x_atlpd, y_atlpd, random_state = 0)

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train_scaled, y_train)
print "Accuracy of K-NN classifier on training set: {:.2f}".format(knn.score(x_train_scaled, y_train))
print "Accuracy of K-NN classifier on test set: {:.2f}".format(knn.score(x_test_scaled, y_test))

y_predicted = knn.predict(x_test)
df_cm = pd.DataFrame(confusion_matrix(y_test, y_predicted))
plt.figure(figsize = (6, 4))
conf_heat = sns.heatmap(df_cm, annot = True, fmt = "d", cmap = "YlGnBu", xticklabels = target_names, yticklabels = target_names)

print "Accuracy: {:.2f}".format(accuracy_score(y_test, y_predicted))
print "Precision: {:.2f}".format(precision_score(y_test, y_predicted))
print "Recall: {:.2f}".format(recall_score(y_test, y_predicted))
print "F1: {:.2f}".format(f1_score(y_test, y_predicted))
```

## Logistic Regression for Lethalness

```python
feature_names = ["Hour", "DoW", "NPU Num"]
x_atlpd = atlpd[feature_names]
y_atlpd = atlpd["Lethalness Num"]
target_names = atlpd["Lethalness"].cat.categories

x_train, x_test, y_train, y_test = train_test_split(x_atlpd, y_atlpd, random_state=0)

clf = LogisticRegression(C = 100, solver = "lbfgs", multi_class = "auto")
clf.fit(x_train, y_train)
print "Accuracy of Logistic regression classifier on training set: {:.2f}".format(clf.score(x_train, y_train))
print "Accuracy of Logistic regression classifier on test set: {:.2f}".format(clf.score(x_test, y_test))

y_predicted = clf.predict(x_test)
df_cm = pd.DataFrame(confusion_matrix(y_test, y_predicted))
plt.figure(figsize = (6, 4))
conf_heat = sns.heatmap(df_cm, annot = True, fmt = "d", cmap = "YlGnBu", xticklabels = target_names, yticklabels = target_names)

print "Accuracy: {:.2f}".format(accuracy_score(y_test, y_predicted))
print "Precision: {:.2f}".format(precision_score(y_test, y_predicted))
print "Recall: {:.2f}".format(recall_score(y_test, y_predicted))
print "F1: {:.2f}".format(f1_score(y_test, y_predicted))
```

## Logistic Regression for Crime

```python
feature_names = ["Hour", "DoW", "NPU Num"]
x_atlpd = atlpd[feature_names]
y_atlpd = atlpd["UCR Code"]
target_names = atlpd["UCR Literal"].cat.categories

x_train, x_test, y_train, y_test = train_test_split(x_atlpd, y_atlpd, random_state=0)

clf = LogisticRegression(C = 20, solver = "lbfgs", multi_class = "auto")
clf.fit(x_train, y_train)
print "Accuracy of Logistic regression classifier on training set: {:.2f}".format(clf.score(x_train, y_train))
print "Accuracy of Logistic regression classifier on test set: {:.2f}".format(clf.score(x_test, y_test))

y_predicted = clf.predict(x_test)

print "Micro-averaged precision = {:.2f} (treat instances equally)".format(precision_score(y_test, y_predicted, average = "micro"))
print "Macro-averaged precision = {:.2f} (treat classes equally)".format(precision_score(y_test, y_predicted, average = "macro"))
print "Micro-averaged F1 = {:.2f} (treat instances equally)".format(f1_score(y_test, y_predicted, average = "micro"))
print "Macro-averaged F1 = {:.2f} (treat classes equally)".format(f1_score(y_test, y_predicted, average = "macro"))

df_cm = pd.DataFrame(confusion_matrix(y_test, y_predicted))
plt.figure(figsize = (12, 6))
conf_heat = sns.heatmap(df_cm, annot = True, fmt = "d", cmap = "YlGnBu", xticklabels = target_names, yticklabels = target_names)
```

## Decision Tree for Crime

```python
atlpd_extended = atlpd.copy()
atlpd_extended = pd.concat([atlpd_extended, pd.get_dummies(atlpd['Shift Occurrence'])], axis = 1)
atlpd_extended = pd.concat([atlpd_extended, pd.get_dummies(atlpd['Lethalness'])], axis = 1)
atlpd_extended = pd.concat([atlpd_extended, pd.get_dummies(atlpd['Day of Week'])], axis = 1)

feature_names = ["Year", "Month", "Hour", "NPU Num"]
feature_names.extend(pd.get_dummies(atlpd['Shift Occurrence']).columns)
feature_names.extend(pd.get_dummies(atlpd['Lethalness']).columns)
feature_names.extend(pd.get_dummies(atlpd['Day of Week']).columns)
feature_names.remove("Unknown")
x_atlpd = atlpd_extended[feature_names]
y_atlpd = atlpd_extended["UCR Code"]
target_names = atlpd_extended["UCR Literal"].cat.categories

x_train, x_test, y_train, y_test = train_test_split(x_atlpd, y_atlpd, random_state=0)

clf = DecisionTreeClassifier(max_depth = 4).fit(x_train, y_train)
print "Accuracy of Decision Tree classifier on training set: {:.2f}".format(clf.score(x_train, y_train))
print "Accuracy of Decision Tree classifier on test set: {:.2f}".format(clf.score(x_test, y_test))

y_predicted = clf.predict(x_test)

print "Micro-averaged precision = {:.2f} (treat instances equally)".format(precision_score(y_test, y_predicted, average = "micro"))
print "Macro-averaged precision = {:.2f} (treat classes equally)".format(precision_score(y_test, y_predicted, average = "macro"))
print "Micro-averaged F1 = {:.2f} (treat instances equally)".format(f1_score(y_test, y_predicted, average = "micro"))
print "Macro-averaged F1 = {:.2f} (treat classes equally)".format(f1_score(y_test, y_predicted, average = "macro"))

df_cm = pd.DataFrame(confusion_matrix(y_test, y_predicted))
plt.figure(figsize = (12, 6))
conf_heat = sns.heatmap(df_cm, annot = True, fmt = "d", cmap = "YlGnBu", xticklabels = target_names, yticklabels = target_names)

graph = graphviz.Source(export_graphviz(clf, out_file = None, feature_names = feature_names, class_names = target_names, filled = True, impurity = False))
display(SVG(graph.pipe(format = "svg")))
```

***


# About the data...


## Shape of the DataFrame

```python
atlpd.shape
```

## Types of the columns

```python
atlpd.info()
```
