import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os

# ===========================
# 读取 JSON
# ===========================
with open("/workspace/sparsene/examples/src_fp32/dtc/testbed/scripts/motivation_lb/matrix_status.json", "r", encoding="utf-8") as f:
    matrix_status = json.load(f)

for item in matrix_status:
    item["filename"] = os.path.basename(item["filename"])

matrix_status = {
    d["filename"]: {k: v for k, v in d.items() if k != "filename"}
    for d in matrix_status
}

with open("/workspace/sparsene/examples/src_fp32/dtc/testbed/scripts/motivation_lb/results.json", "r", encoding="utf-8") as f:
    results = json.load(f)["results"]

# ===========================
# 整理成 DataFrame
# ===========================
data = []
for r in results:
    if r["normalbinding"] == "" or r["multibinding"] == "" or r["strictlb"] == "":
        continue
    fname = os.path.basename(r["filename"].strip())
    if fname in matrix_status:
        feats = matrix_status[fname].copy()

        times = {
            "normalbinding": float(r["normalbinding"]),
            "multibinding": float(r["multibinding"]),
            "strictlb": float(r["strictlb"])
        }
        best_method = min(times, key=times.get)

        feats["filename"] = fname
        feats["best"] = best_method
        data.append(feats)

df = pd.DataFrame(data)

# 特征 + 标签
X = df[["nrow", "ncol", "nnz", "nnz_max", "nnz_min", "nnz_mean", "nnz_std"]]
y = df["best"]

# ===========================
# 划分训练/测试集
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
# 多模型训练与预测
# ===========================
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", class_weight="balanced")
}

# 训练其他模型
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("="*50)
    print(f"模型: {name}")
    print("准确率:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# ===========================
# 训练 XGBoost（需要 LabelEncoder）
# ===========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric='mlogloss'
)
xgb_model.fit(X_train_enc, y_train_enc)
y_pred_enc = xgb_model.predict(X_test_enc)
y_pred_labels = le.inverse_transform(y_pred_enc)
y_test_labels = le.inverse_transform(y_test_enc)

print("="*50)
print("模型: XGBoost")
print("准确率:", accuracy_score(y_test_labels, y_pred_labels))
print(classification_report(y_test_labels, y_pred_labels))

# ===========================
# 对新矩阵预测
# ===========================
new_matrix = {
    "nrow": 929901,
    "ncol": 303645,
    "nnz": 4020731,
    "nnz_max": 5,
    "nnz_min": 2,
    "nnz_mean": 4.323827,
    "nnz_std": 0.468915
}
X_new = pd.DataFrame([new_matrix])

print("="*50)
print("新矩阵预测结果：")
for name, model in models.items():
    print(f"{name} 预测最优方法:", model.predict(X_new)[0])

# XGBoost
y_new_pred_enc = xgb_model.predict(X_new)
y_new_pred_label = le.inverse_transform(y_new_pred_enc)
print(f"XGBoost 预测最优方法: {y_new_pred_label[0]}")
