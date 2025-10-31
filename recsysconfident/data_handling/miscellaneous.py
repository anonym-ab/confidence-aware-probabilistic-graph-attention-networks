import pandas as pd


def keep_users_any_r_higher_than(df, user_col: str, rating_col: str, threshold: float):
    filtered_df = df.groupby(user_col).filter(lambda x: (x[rating_col] >= threshold).any())
    return filtered_df.reset_index(drop=True)

def filter_out_users_less_than_k_inter(df: pd.DataFrame, user_col: str, min_iterations: int):
    df = df.copy()
    df = df.groupby(user_col).filter(
        lambda x: len(x) >= min_iterations)  # ensure there is at least k ratings per user

    return df



