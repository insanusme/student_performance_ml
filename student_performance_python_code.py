# 1. 匯入套件
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 2. 讀取資料
df = pd.read_csv('xAPI-Edu-Data.csv')

df = df.drop_duplicates() #刪除所有重複列（保留第一次出現的）

# 3. 檢查資料
print(df.info())
print("重複筆數：",df.duplicated().sum())
print("缺失值總表：\n",df.isnull().sum())



# 類別資料編碼，將每一類別對應到一個整數
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
    
# 5. 顯示相關係數
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# 6.直方圖
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='raisedhands', hue='Class', kde=True, multiple='stack')
plt.title("Raised Hands Distribution by Class")
plt.show()

# 7.盒方圖
plt.figure(figsize=(8, 5))
sns.boxplot(x='Class', y='Discussion', data=df)
plt.title("Discussion Boxplot by Class")
plt.show()

# 8.所有數值型特徵的直方圖
num_features = df.select_dtypes(include=['int64', 'float64']).columns
df[num_features].hist(bins=20, figsize=(12, 10))
plt.tight_layout()
plt.show()

# 9./標籤分離
X = df.drop('Class', axis=1)
y = df['Class']

# 10.切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11.模型訓練與預測
# (1) Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# (2) KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# (3) Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 12.評估函式
def evaluate(model_name, y_true, y_pred):
    print(f"=== {model_name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\n")

# 13.顯示結果
evaluate("Random Forest", y_test, y_pred_rf)
evaluate("KNN", y_test, y_pred_knn)
evaluate("Logistic Regression", y_test, y_pred_lr)