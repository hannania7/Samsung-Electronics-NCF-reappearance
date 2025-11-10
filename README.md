# 🧠 NCF-based Recommender System (Santander Product Recommendation)

## 📌 Overview
TensorFlow 기반 **Neural Collaborative Filtering (NCF)** 추천 모델 구현 프로젝트입니다.  
Kaggle Santander Product Recommendation 데이터를 활용하여,  
삼성전자(폐쇄망) 실무에서 사용한 추천모델 파이프라인(NCF + BPR Loss)을 로컬 환경에서 재현했습니다.

- Dataset: [Santander Product Recommendation (Kaggle)](https://www.kaggle.com/datasets/padmanabhanporaiyar/santander-product-recommendation-parquet-data)
- Evaluation: Leave-One-Out (LOO)
- Metric: HR@5
- HR@5 = **0.41** (샘플 유저 16명 기준)

---

## ⚙️ Architecture
```text
NCF.py
├── get_long_df() # 데이터 로드 및 wide → long 변환
├── preprocess_long_df() # 유저/아이템 ID 매핑
├── build_user_sequences() # 유저별 시퀀스 생성
├── make_train_test() # LOO 방식 분할
├── NCF # Embedding + MLP + BPR Loss
├── train_model() # HR@5 기반 학습 루프
└── save_recommendations() # 결과 CSV 저장
```
