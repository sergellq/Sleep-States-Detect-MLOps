import pandas as pd
from scipy.signal import find_peaks


def predict_peaks(predict):
    """
    find peaks from formated time series dataframe

    Args:
        predict: pd.df with columns series_id, date, time, step, event, score
    Returs:
        pd.df with columns row_id, series_id, step, event, score
    """
    list_df = []
    for series_id, df in predict.groupby("series_id"):
        for event in ["onset", "wakeup"]:
            values_step = df["step"].values
            if event == "onset":
                values_score = -df["score"].values
            else:
                values_score = df["score"].values

            peak_idx = find_peaks(values_score, height=0.0, distance=8)[0]
            df_peak = pd.DataFrame(values_step[peak_idx], columns=["step"])
            df_peak["series_id"] = series_id
            df_peak["event"] = event
            df_peak["score"] = values_score[peak_idx]
            list_df.append(df_peak)

    df_sub = pd.concat(list_df)
    df_sub = (
        df_sub.sort_values("score", ascending=False).groupby("event").head(100000)
    )  # avoid Submission Scoring Error
    df_sub = df_sub.sort_values(["series_id", "step"]).reset_index(drop=True)
    df_sub = df_sub[["series_id", "step", "event", "score"]].reset_index(names="row_id")
    return df_sub
