import pandas as pd


def format_as_dataframe(df):
    df = pd.DataFrame(df)
    df.fillna(0, inplace=True)
    df = df.astype(int)
    df.loc['Totals'] = df.sum(numeric_only=True, axis=0)
    return df


def create_misclassification_df(selected_df, model_accuracies):
    misclassification_dict = {}
    for i in selected_df.columns:
        misclassification_dict[i] = (selected_df[i] * (1 - model_accuracies[i])).round().astype(int)

    return pd.DataFrame(misclassification_dict)