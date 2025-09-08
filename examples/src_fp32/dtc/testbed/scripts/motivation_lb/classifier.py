import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# 读取 JSON
with open("/workspace/sparsene/examples/src_fp32/dtc/testbed/scripts/motivation_lb/matrix_status.json", "r", encoding="utf-8") as f:
    matrix_status = json.load(f)  # {filename: {...features...}}
for item in matrix_status:
    item["filename"] = os.path.basename(item["filename"])
result = {
    d["filename"]: {k: v for k, v in d.items() if k != "filename"}
    for d in matrix_status
}
matrix_status = result

with open("/workspace/sparsene/examples/src_fp32/dtc/testbed/scripts/motivation_lb/results.json", "r", encoding="utf-8") as f:
    results = json.load(f)  # [{"filename": "...", "normalbinding": "...", "multibinding": "...", "strictlb": "..."}]
    results = results["results"]

# 整理成 DataFrame
data = []
for r in results:
    if r["normalbinding"] == "" or r["multibinding"] == "" or r["strictlb"] == "":
        continue
    fname = os.path.basename(r["filename"].strip())
    if fname in matrix_status:
        feats = matrix_status[fname].copy()

        # 找最优方法
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

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型（决策树，可以换成 RandomForest / XGBoost）
clf = DecisionTreeClassifier(max_depth=6, random_state=42)
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
print("准确率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

