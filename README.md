# 🧠 NCF-based Recommender System (Santander Product Recommendation)

## 📌 Overview
TensorFlow 기반 유저와 아이디간의 관계로 학습하는 **Neural Collaborative Filtering (NCF)** 추천 모델 구현 프로젝트입니다.  
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


📊 Additional Notes
**통계 기반 추천(Popularity / Co-occurrence 등)**은
현업 파이프라인에서는 함께 사용되었으나,
본 프로젝트에서는 데이터 용량이 매우 커 학습 및 실험 효율을 위해 제외하였습니다.

대신 본 구현에서는 Neural Collaborative Filtering (NCF) 모델을 중심으로
Embedding + MLP + BPR Loss + negative_sampling = 1 구조를 재현하여 핵심 추천 로직을 복원하였습니다.

추후 데이터셋을 경량화하거나 샘플링을 적용하면,
통계 기반 추천을 결합한 Hybrid Recommendation System 형태로 확장 가능합니다.

추가로 BERT4REC은 삼성전자 VOC데이터 특성상 아이템 간의 문맥 정보는 없다고 생각하어 NCF로 확정하였습니다.