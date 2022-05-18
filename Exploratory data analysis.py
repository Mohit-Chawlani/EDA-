
# Load the required libraries
import pandas as pd
import numpy as np
import seaborn as sns

# Load the data
data = pd.read_csv("D:\heart.csv")

# View the data
data.head()
print(data.head)

# Basic information

data.info()

# Describe the data

data.describe()

# Find null values

data.isnull().sum()

#line plot
import seaborn as sns
import matplotlib.pyplot as plt

# loading dataset
import pandas as pd
data = pd.read_csv("D:\heart.csv")

# draw lineplot
sns.lineplot(x="age", y="trtbps", data=data)

# Removing the spines
sns.despine()
plt.show()

#Scatter Plot
import seaborn as sns
import matplotlib.pyplot as plt

# loading dataset
import pandas as pd
data = pd.read_csv("D:\heart.csv")

# draw Sctter Plot
sns.Scatterplot(x="age", y="trtbps", data=data)

# Removing the spines
sns.despine()
plt.show()

# Bar Plot
# from seaborn library
data = sns.load_dataset("D:\heart.csv")

# class v / s fare barplot
sns.barplot(x='age', y='trtbps', data=data)

# Show the plot
plt.show()

#correlation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("D:\heart.csv")
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()

# Linear regression
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

data = pd.read_csv("D:\heart.csv")
x = data["age"]
y = data["trtbps"]

slope, intercept, r, p, std_err = stats.linregress(x, y)


def myfunc(x):
    return slope * x + intercept


mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

from sklearn import linear_model
import pandas as pd
data =pd.read_csv("D:\heart.csv")
X = data[['age']]
y = data['trtbps']

regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.coef_)

