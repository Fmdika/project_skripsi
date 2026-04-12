import pandas as pd

def preprocess_daily(df, date_col, value_cols):
    """
    Preprocessing data harian (IHSG, kurs)
    - parsing tanggal
    - sorting
    - seleksi kolom
    """
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df[[date_col] + value_cols]

    return df


def preprocess_monthly_to_daily(df, date_col, value_col):
    """
    Mengubah data bulanan menjadi harian
    dengan forward fill
    """
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    df_daily = df.resample("D").ffill()
    df_daily = df_daily.reset_index()

    return df_daily

def merge_datasets(base_df, dfs, on="date"):
    """
    Menggabungkan beberapa dataset
    berdasarkan kolom tanggal
    """
    df_final = base_df.copy()

    for df in dfs:
        df_final = df_final.merge(df, on=on, how="left")

    return df_final
