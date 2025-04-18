import pandas as pd


def correlation_analysis(X: pd.DataFrame, y: pd.DataFrame):
    correlation = X.corrwith(y, method="pearson")
    print(correlation)
