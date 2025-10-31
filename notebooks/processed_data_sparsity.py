import json

import pandas as pd

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


def get_description(df, user_col, item_col, name):
    n_users = len(df[user_col].unique())
    n_items = len(df[item_col].unique())

    df_sparsity = 1 - len(df) / (n_users * n_items)
    return {
        "dataset": name,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": len(df),
        "sparsity": df_sparsity,
    }


if __name__ == "__main__":

    processed_dfs_names = ['amazon-movies-tvs', 'ml-1m', 'netflix-prize']
    df_descs = []
    for dataset_name in processed_dfs_names:
        base_uri = f"../runs/data_splits/{dataset_name}/0/"
        fit_df = pd.read_csv(f"{base_uri}/ratings.fit.csv")
        eval_df = pd.read_csv(f"{base_uri}/ratings.val.csv")
        test_df = pd.read_csv(f"{base_uri}/ratings.test.csv")
        df = pd.concat([fit_df, eval_df, test_df])



        df_info = DatasetInfo(**json.load(open(f'../data/{dataset_name}/info.json', 'r')),
                            database_name=dataset_name,
                            root_uri="..")

        df_desc = get_description(df, df_info.user_col, df_info.item_col, dataset_name)
        df_desc['train_ratio'] = len(fit_df) / len(df)
        df_desc['eval_ratio'] = len(eval_df) / len(df)
        df_desc['test_ratio'] = len(test_df) / len(df)

        df_descs.append(df_desc)

    descs_df = pd.DataFrame(df_descs)

    str_table = descs_df.to_latex(
            label="label",
            caption="caption",
            index=False,
            escape=False,
            column_format="c" * len(descs_df.columns)  # Center align columns
        )
    print(str_table)
