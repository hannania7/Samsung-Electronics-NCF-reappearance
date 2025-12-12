import pandas as pd
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm
import kagglehub
import pyarrow.dataset as ds
import pyarrow as pa

from xgboost import XGBClassifier  # ← 여기만 바뀜


# ===== 공통 시드 =====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# ===== 1. 데이터 준비 =====
def get_long_df() -> pd.DataFrame:
    path = kagglehub.dataset_download("padmanabhanporaiyar/santander-product-recommendation-parquet-data")
    print("Path to dataset files:", path)

    base = r"C:\Users\user\.cache\kagglehub\datasets\padmanabhanporaiyar\santander-product-recommendation-parquet-data\versions\1\paraquet files"

    dataset = ds.dataset(base, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()

    product_cols = [c for c in df.columns if c.startswith("ind_") and c.endswith("_ult1")]
    id_cols = ["ncodpers", "fecha_dato"]

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=product_cols,
        var_name="product",
        value_name="value"
    )

    long_df = long_df[long_df["value"] == 1].reset_index(drop=True)
    return long_df


# ===== 2. 5개 이상 가진 유저만 남기기 + id 매핑 =====
def preprocess_long_df(long_df: pd.DataFrame):
    cnt_per_user = long_df.groupby("ncodpers")["product"].nunique()
    valid_users = cnt_per_user[cnt_per_user >= 5].index
    long_df = long_df[long_df["ncodpers"].isin(valid_users)].reset_index(drop=True)

    user2id = {u: i for i, u in enumerate(long_df["ncodpers"].unique())}
    item2id = {p: i for i, p in enumerate(long_df["product"].unique())}
    long_df["user_id"] = long_df["ncodpers"].map(user2id)
    long_df["item_id"] = long_df["product"].map(item2id)

    return long_df, user2id, item2id


# ===== 3. 유저별 시퀀스 만들기 =====
def build_user_sequences(long_df: pd.DataFrame):
    user_pos_items = defaultdict(list)
    has_date = "fecha_dato" in long_df.columns

    if has_date:
        long_df = long_df.sort_values(["user_id", "fecha_dato"])
    else:
        long_df = long_df.sort_values(["user_id", "item_id"])

    for row in long_df.itertuples():
        user_pos_items[row.user_id].append(
            (row.item_id, getattr(row, "fecha_dato", None))
        )
    return user_pos_items


# ===== 4. 유저 샘플링 =====
def sample_users(user_pos_items, max_users=500):
    users_5plus = [u for u, items in user_pos_items.items() if len(items) >= 5]
    sampled_users = random.sample(users_5plus, min(max_users, len(users_5plus)))
    return sampled_users


# ===== 5. LOO로 train/test 만들기 =====
def make_train_test(user_pos_items, sampled_users, max_train=500, max_test=100):
    train_pairs = []
    test_pairs = []

    for u in sampled_users:
        items = user_pos_items[u]
        if len(items) == 1:
            test_pairs.append((u, items[0][0]))
        else:
            for it, _ in items[:-1]:
                train_pairs.append((u, it))
            test_pairs.append((u, items[-1][0]))

    if len(train_pairs) > max_train:
        train_pairs = random.sample(train_pairs, max_train)
    if len(test_pairs) > max_test:
        test_pairs = random.sample(test_pairs, max_test)

    return train_pairs, test_pairs


# ===== 6. 마스킹/보유 set =====
def build_pos_sets(train_pairs, test_pairs):
    user_train_pos_set = defaultdict(set)
    for u, it in train_pairs:
        user_train_pos_set[u].add(it)

    user_all_pos_set = defaultdict(set)
    for u, it in train_pairs:
        user_all_pos_set[u].add(it)
    for u, it in test_pairs:
        user_all_pos_set[u].add(it)

    return user_train_pos_set, user_all_pos_set


# ===== 7. XGBoost용 학습 데이터 만들기 =====
def build_xgb_training_data(train_pairs, sampled_users, num_items, user_all_pos_set, neg_ratio=3):
    X_user = []
    X_item = []
    y = []

    all_items = set(range(num_items))

    for (u, pos_it) in train_pairs:
        # positive
        X_user.append(u)
        X_item.append(pos_it)
        y.append(1)

        # negative
        can_neg = list(all_items - user_all_pos_set[u])
        if not can_neg:
            continue
        n_to_sample = min(neg_ratio, len(can_neg))
        neg_items = random.sample(can_neg, n_to_sample)
        for ni in neg_items:
            X_user.append(u)
            X_item.append(ni)
            y.append(0)

    X = np.column_stack([X_user, X_item])
    y = np.array(y, dtype=int)
    return X, y


# ===== 8. HR@k (XGBoost) =====
def hitrate_at_k_xgb(model, test_pairs, user_train_pos_set, num_items, k=5):
    if not test_pairs:
        return 0.0

    hits = 0
    for (u, true_item) in test_pairs:
        items = np.arange(num_items, dtype=int)
        users = np.full_like(items, u)
        X = np.column_stack([users, items])

        probs = model.predict_proba(X)[:, 1]

        # train에서 본 아이템은 제외
        for seen in user_train_pos_set[u]:
            probs[seen] = -1e9

        topk_idx = np.argpartition(probs, -k)[-k:]
        topk_idx = topk_idx[np.argsort(probs[topk_idx])][::-1]

        if true_item in topk_idx:
            hits += 1

    return hits / len(test_pairs)


# ===== 9. 추천 CSV =====
def recommend_topk_from_owned_xgb(model, user_id, user_all_pos_set, num_items, k=5):
    owned_items = list(user_all_pos_set.get(user_id, []))
    if len(owned_items) < k:
        return None

    items = np.arange(num_items, dtype=int)
    users = np.full_like(items, user_id)
    X = np.column_stack([users, items])
    probs = model.predict_proba(X)[:, 1]

    owned_scores = [(it, float(probs[it])) for it in owned_items]
    owned_scores.sort(key=lambda x: x[1], reverse=True)
    return [it for it, _ in owned_scores[:k]]


def save_recommendations_xgb(model, test_pairs, user_all_pos_set, num_items,
                             max_csv_users=50, out_path="recommendations_test_users_from_xgboost.csv"):
    rows = []
    test_users = {u for (u, _) in test_pairs}
    test_users = list(test_users)[:max_csv_users]
    for u in test_users:
        top_items = recommend_topk_from_owned_xgb(model, u, user_all_pos_set, num_items, k=5)
        if top_items is None:
            continue
        rows.append({"user_id": int(u), "items": str(top_items)})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"추천 저장 완료: {out_path}")

# ===== (추가) 유저별 hit(0/1) 뽑기 =====
def hit_per_user_xgb(model, test_pairs, user_train_pos_set, num_items, k=5):
    hits = []
    for (u, true_item) in test_pairs:
        items = np.arange(num_items, dtype=int)
        users = np.full_like(items, u)
        X = np.column_stack([users, items])
        probs = model.predict_proba(X)[:, 1]

        for seen in user_train_pos_set[u]:
            probs[seen] = -1e9

        topk = np.argpartition(probs, -k)[-k:]
        topk = topk[np.argsort(probs[topk])][::-1]
        hits.append({"user_id": int(u), "hit": int(true_item in topk)})
    return pd.DataFrame(hits)

# ===== (추가) 유저별 hit 저장 =====
def save_hits_xgb(model, test_pairs, user_train_pos_set, num_items, k=5, out_path="hits_xgb.csv"):
    df_hits = hit_per_user_xgb(model, test_pairs, user_train_pos_set, num_items, k=k)
    df_hits.to_csv(out_path, index=False)
    print(f"XGB hits saved: {out_path}, HR@{k}={df_hits['hit'].mean():.4f}")

# ===== main =====
def main():
    set_seed(42)

    # 1. 데이터 읽기
    long_df = get_long_df()

    # 2. 전처리
    long_df, user2id, item2id = preprocess_long_df(long_df)
    num_users = len(user2id)
    num_items = len(item2id)
    print(f"num_users={num_users}, num_items={num_items}")

    # 3. 유저 시퀀스
    user_pos_items = build_user_sequences(long_df)

    # 4. 유저 샘플링
    sampled_users = sample_users(user_pos_items, max_users=500)

    # 5. LOO split
    train_pairs, test_pairs = make_train_test(user_pos_items, sampled_users,
                                              max_train=500, max_test=100)

    # 6. set들
    user_train_pos_set, user_all_pos_set = build_pos_sets(train_pairs, test_pairs)

    # 7. XGBoost 학습 데이터
    X, y = build_xgb_training_data(train_pairs, sampled_users, num_items, user_all_pos_set, neg_ratio=3)
    print("train shape:", X.shape, y.shape)

    # 8. 모델 정의 & 학습
    # 파라미터는 대략만 넣어둠. 데이터 많아지면 n_estimators 줄이거나 max_depth 조절.
    xgb = XGBClassifier(
        random_state=42
    )
    xgb.fit(X, y)

    # 9. 평가
    hr5 = hitrate_at_k_xgb(xgb, test_pairs, user_train_pos_set, num_items, k=5)
    print(f"HR@5 = {hr5:.4f}")

    # 10. 추천 저장
    save_recommendations_xgb(xgb, test_pairs, user_all_pos_set, num_items)

    save_hits_xgb(xgb, test_pairs, user_train_pos_set, num_items, k=5, out_path="hits_xgb.csv")


if __name__ == "__main__":
    main()
