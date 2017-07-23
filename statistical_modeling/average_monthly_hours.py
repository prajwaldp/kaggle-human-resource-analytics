# checking out the distribution of the `average_monthly_hours` column

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("cleaned_df.csv")
mean = np.mean(df["average_monthly_hours"])
std = np.std(df["average_monthly_hours"])


# Function to compute the ECDF
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y


# trying out a normal distribution
samples = np.random.normal(mean, std, size=10000)
x, y = ecdf(df["average_monthly_hours"])
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel("Average Monthly Hours")
plt.ylabel("CDFs")
plt.show()


# trying out an exponential distribution
samples = np.random.exponential(mean, size=10000)
x, y = ecdf(df["average_monthly_hours"])
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel("Average Monthly Hours")
plt.ylabel("CDFs")
plt.show()
