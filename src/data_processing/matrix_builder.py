import pandas as pd
from scipy.sparse import csr_matrix


def build_user_item_matrix(
    ratings: pd.DataFrame,
) -> tuple[csr_matrix, list[int], list[int]]:
    users = sorted(ratings["UserID"].astype(int).unique().tolist())
    items = sorted(ratings["MovieID"].astype(int).unique().tolist())
    user_positions = {user_id: index for index, user_id in enumerate(users)}
    item_positions = {item_id: index for index, item_id in enumerate(items)}
    rows = ratings["UserID"].map(user_positions).to_numpy()
    columns = ratings["MovieID"].map(item_positions).to_numpy()
    values = ratings["Rating"].astype(float).to_numpy()
    matrix = csr_matrix((values, (rows, columns)), shape=(len(users), len(items)))
    return matrix, users, items
