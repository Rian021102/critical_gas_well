import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def eda_train(X_train, y_train):
    df_train = pd.concat([X_train, y_train], axis=1)
# Check for Missing and Infinite Valuesprint('Checking Missing Values: ', df_train.isna().sum())
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
    df_train.dropna(inplace=True)  # Drop rows with NaN values# Select numeric columns
    numeric_columns = df_train.select_dtypes(include=['float64', 'int64']).columns
    df_train_num = df_train[numeric_columns]
# Histogram
    plt.figure()
    df_train_num.hist(figsize=(20, 15))
    plt.show()
    plt.close()
# Plot correlation matrix
    plt.figure()
    corr = df_train_num.corr()
    sns.heatmap(corr, annot=True)
    plt.show()
    plt.close()

    #plot boxplot
    plt.figure()
    sns.boxplot(data=df_train_num)
    plt.show()
    plt.close()

    return df_train