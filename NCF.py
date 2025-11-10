import pandas as pd
import numpy as np
import tensorflow as tf
from collections import defaultdict
import random
from tqdm import tqdm
import kagglehub
import pyarrow.dataset as ds
import pyarrow as pa

# ===== 공통 시드 =====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ===== 1. 데이터 준비 (여긴 네가 채우는 자리) =====
def get_long_df() -> pd.DataFrame:

    # Download latest version
    path = kagglehub.dataset_download("padmanabhanporaiyar/santander-product-recommendation-parquet-data")

    print("Path to dataset files:", path)

    base = r"C:\Users\user\.cache\kagglehub\datasets\padmanabhanporaiyar\santander-product-recommendation-parquet-data\versions\1\paraquet files"

    # 폴더 전체를 하나의 dataset으로 본다
    dataset = ds.dataset(base, format="parquet")

    # 한 번에 테이블로 뽑기 (← 여기서 이미 확장타입을 한 번만 처리함)
    table = dataset.to_table()

    # pandas로 변환
    df = table.to_pandas()

    # 2. 상품 컬럼 자동으로 찾기
    #    보통 상품 컬럼들은 ind_로 시작하고 _ult1로 끝남
    product_cols = [c for c in df.columns if c.startswith("ind_") and c.endswith("_ult1")]

    # 3. 필요한 기본 컬럼 지정
    #    ncodpers = 고객 ID, fecha_dato = 날짜(월)
    id_cols = ["ncodpers", "fecha_dato"]

    # 혹시 날짜가 없으면 이렇게 해도 됨:
    # id_cols = ["ncodpers"]

    # 4. wide → long 으로 녹이기
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=product_cols,
        var_name="product",
        value_name="value"
    )

    # 5. 실제로 보유(=1)한 것만 남기기
    long_df = long_df[long_df["value"] == 1].reset_index(drop=True)
    return long_df


# ===== 2. 5개 이상 가진 유저만 남기기 + id 매핑 =====
def preprocess_long_df(long_df: pd.DataFrame):
    # 5개 이상
    cnt_per_user = long_df.groupby("ncodpers")["product"].nunique()
    valid_users = cnt_per_user[cnt_per_user >= 5].index
    long_df = long_df[long_df["ncodpers"].isin(valid_users)].reset_index(drop=True)

    # id 매핑
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


# ===== 4. 유저 샘플링 (사이즈 줄이기) =====
def sample_users(user_pos_items, max_users=100):
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

    # 사이즈 줄이기
    if len(train_pairs) > max_train:
        train_pairs = random.sample(train_pairs, max_train)
    if len(test_pairs) > max_test:
        test_pairs = random.sample(test_pairs, max_test)

    return train_pairs, test_pairs


# ===== 6. 마스킹/보유 set 만들기 =====
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


# ===== 7. NCF 모델 =====
class NCF(tf.keras.Model):
    def __init__(self, num_users, num_items, emb_dim=64, mlp_layers=(256, 128, 64), dropout=0.1):
        super().__init__()
        self.user_emb = tf.keras.layers.Embedding(num_users, emb_dim)
        self.item_emb = tf.keras.layers.Embedding(num_items, emb_dim)

        layers = []
        for h in mlp_layers:
            layers.append(tf.keras.layers.Dense(h, activation="relu"))
            layers.append(tf.keras.layers.Dropout(dropout))
        self.mlp = tf.keras.Sequential(layers)
        self.out = tf.keras.layers.Dense(1)

    def call(self, user_ids, item_ids, training=False):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = tf.concat([u, i], axis=-1)
        x = self.mlp(x, training=training)
        x = self.out(x)
        return tf.squeeze(x, axis=-1)

    def full_score(self, user_id, num_items):
        user_id = tf.constant([user_id], dtype=tf.int32)
        u = self.user_emb(user_id)
        all_items = tf.range(num_items, dtype=tf.int32)
        i = self.item_emb(all_items)
        u = tf.repeat(u, repeats=num_items, axis=0)
        x = tf.concat([u, i], axis=-1)
        x = self.mlp(x, training=False)
        x = self.out(x)
        x = tf.squeeze(x, axis=-1)
        return tf.nn.softmax(x)


# ===== 8. BPR loss =====
@tf.function
def bpr_loss(pos_scores, neg_scores):
    diff = pos_scores - neg_scores
    return -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(diff) + 1e-8))


# ===== 9. 네거티브 풀 만들기 =====
def build_negative_pool(sampled_users, num_items, user_all_pos_set):
    all_items_set = set(range(num_items))
    neg_pool = {}
    for u in sampled_users:
        remain = list(all_items_set - user_all_pos_set[u])
        neg_pool[u] = remain if remain else list(all_items_set)
    return neg_pool


# ===== 10. HR@5 =====
def hitrate_at_k(model, test_pairs, user_train_pos_set, num_items, k=5):
    if not test_pairs:
        return 0.0
    hits = 0
    for (u, true_item) in test_pairs:
        scores = model.full_score(u, num_items).numpy()
        # train에서 본 것만 막기
        for s in user_train_pos_set[u]:
            scores[s] = -1e9
        topk = np.argpartition(scores, -k)[-k:]
        topk = topk[np.argsort(scores[topk])][::-1]
        if true_item in topk:
            hits += 1
    return hits / len(test_pairs)


# ===== 11. 학습 =====
def train_model(model, train_pairs, test_pairs, user_train_pos_set, neg_pool, num_items,
                lr, epochs, batch_size, patience=5):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
    training_history = []
    best_hr = 0.0
    wait = 0

    for epoch in range(1, epochs + 1):
        random.shuffle(train_pairs)
        total_loss = 0.0

        for start in tqdm(range(0, len(train_pairs), batch_size), desc=f"Epoch {epoch}", leave=False):
            batch = train_pairs[start:start + batch_size]
            u_list, pos_list, neg_list = [], [], []

            for (u, pos) in batch:
                cand = neg_pool[u]
                neg = cand[random.randint(0, len(cand) - 1)]
                u_list.append(u)
                pos_list.append(pos)
                neg_list.append(neg)

            u_tensor = tf.constant(u_list, dtype=tf.int32)
            pos_tensor = tf.constant(pos_list, dtype=tf.int32)
            neg_tensor = tf.constant(neg_list, dtype=tf.int32)

            with tf.GradientTape() as tape:
                pos_scores = model(u_tensor, pos_tensor, training=True)
                neg_scores = model(u_tensor, neg_tensor, training=True)
                loss = bpr_loss(pos_scores, neg_scores)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += float(loss) * len(batch)

        avg_loss = total_loss / len(train_pairs)
        hr5 = hitrate_at_k(model, test_pairs, user_train_pos_set, num_items, k=5)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, HR@5={hr5:.4f}")

        training_history.append({"epoch": epoch, "loss": avg_loss, "hr5": hr5})

        if hr5 > best_hr:
            best_hr = hr5
            wait = 0
            model.save_weights("best_ncf_tf.weights.h5")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping!")
                break

    print("Best HR@5:", best_hr)
    return training_history


# ===== 12. 추천 CSV 만들기 =====
def recommend_topk_from_owned(model, user_id, user_all_pos_set, num_items, k=5):
    owned_items = list(user_all_pos_set.get(user_id, []))
    if len(owned_items) < k:
        return None
    scores = model.full_score(user_id, num_items).numpy()
    owned_scores = [(item_id, float(scores[item_id])) for item_id in owned_items]
    owned_scores.sort(key=lambda x: x[1], reverse=True)
    return [it for it, _ in owned_scores[:k]]


def save_recommendations(model, test_pairs, user_all_pos_set, num_items,
                         max_csv_users=50, out_path="recommendations_test_users_from_NCF.csv"):
    rows = []
    test_users = {u for (u, _) in test_pairs}
    test_users = list(test_users)[:max_csv_users]
    for u in test_users:
        top_items = recommend_topk_from_owned(model, u, user_all_pos_set, num_items, k=5)
        if top_items is None:
            continue
        rows.append({"user_id": int(u), "items": str(top_items)})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"추천 저장 완료: {out_path}")


# ===== 13. 하이퍼파라미터 선택 =====
def choose_hparams(n_samples):
    if n_samples >= 16384:
        return 1024, 0.01, 20
    elif n_samples >= 8192:
        return 512, 0.005, 20
    else:
        return 64, 0.002, 20


# ===== main =====
def main():
    set_seed(42)

    # 1. 데이터 읽기
    long_df = get_long_df()  # 네 환경에 맞게 구현

    # 2. 전처리
    long_df, user2id, item2id = preprocess_long_df(long_df)
    num_users = len(user2id)
    num_items = len(item2id)

    # 3. 유저 시퀀스
    user_pos_items = build_user_sequences(long_df)

    # 4. 유저 샘플링
    sampled_users = sample_users(user_pos_items, max_users=100)

    # 5. LOO split
    train_pairs, test_pairs = make_train_test(user_pos_items, sampled_users,
                                              max_train=500, max_test=100)

    # 6. set들
    user_train_pos_set, user_all_pos_set = build_pos_sets(train_pairs, test_pairs)

    # 7. 하이퍼파라미터
    batch_size, lr, epochs = choose_hparams(len(train_pairs))

    # 8. 모델
    model = NCF(num_users, num_items, emb_dim=64, mlp_layers=(256, 128, 64), dropout=0.1)

    # 9. neg pool
    neg_pool = build_negative_pool(sampled_users, num_items, user_all_pos_set)

    # 10. 학습
    history = train_model(
        model,
        train_pairs,
        test_pairs,
        user_train_pos_set,
        neg_pool,
        num_items,
        lr,
        epochs,
        batch_size,
        patience=5,
    )

    # 11. 추천 저장
    save_recommendations(model, test_pairs, user_all_pos_set, num_items)


if __name__ == "__main__":
    main()
