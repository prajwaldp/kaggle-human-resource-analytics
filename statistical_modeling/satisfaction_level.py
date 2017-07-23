# checking out the distribution of the `satisfaction_level` column

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("cleaned_df.csv")
mean = np.mean(df["satisfaction_level"])
std = np.std(df["satisfaction_level"])


# function to compute the ECDF
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y


# trying out a normal distribution
samples = np.random.normal(mean, std, size=10000)
x, y = ecdf(df["satisfaction_level"])
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel("Satisfaction Levels")
plt.ylabel("CDFs")
plt.show()


# trying out an exponential distribution
samples = np.random.exponential(mean, size=10000)
x, y = ecdf(df["satisfaction_level"])
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel("Satisfaction Levels")
plt.ylabel("CDFs")
plt.show()



def bootstrap_replicates(data, func):
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)


bs_replicates_mean = np.empty(100000)
bs_replicates_var = np.empty(100000)
bs_replicates_std = np.empty(100000)
bs_replicates_median = np.empty(100000)


for i in range(100000):
    bs_replicates_mean[i] = bootstrap_replicates(
        df["satisfaction_level"], np.mean)
    bs_replicates_var[i] = bootstrap_replicates(
        df["satisfaction_level"], np.var)
    bs_replicates_std[i] = bootstrap_replicates(
        df["satisfaction_level"], np.std)
    bs_replicates_median[i] = bootstrap_replicates(
        df["satisfaction_level"], np.median)


plt.hist(bs_replicates_mean, normed=True)
plt.xlabel("Mean Satisfaction levels")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_var, normed=True)
plt.xlabel("Variance of Satisfaction levels")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_std, normed=True)
plt.xlabel("Standard Deviation of Satisfaction levels")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_median, normed=True)
plt.xlabel("Median of Satisfaction levels")
plt.ylabel("PDF")
plt.show()

conf_int_mean = np.percentile(bs_replicates_mean, [2.5, 97.5])
print("Mean:  {:.2f} and {:.2f}".format(
    conf_int_mean[0], conf_int_mean[1]))

conf_int_var = np.percentile(bs_replicates_var, [2.5, 97.5])
print("Variance: {:.2f} and {:.2f}".format(
    conf_int_var[0], conf_int_var[1]))

conf_int_std = np.percentile(bs_replicates_std, [2.5, 97.5])
print("Standard Deviation: {:.2f} and {:.2f}".format(
    conf_int_std[0], conf_int_std[1]))

conf_int_median = np.percentile(bs_replicates_median, [2.5, 97.5])
print("Median: {:.2f} and {:.2f}".format(
    conf_int_median[0], conf_int_median[1]))
