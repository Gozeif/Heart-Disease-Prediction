# 1. تثبيت المكاتب اللازمة (لو مش عندك)
# !pip install xgboost shap joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- ملاحظة: تأكدي أن x_train, x_test, y_train, y_test جاهزين عندك ---

# 2. تعريف الموديلات بـ Hyperparameters احترافية
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, eval_metric='logloss')
}

results = {}

# 3. حلقة التدريب والتقييم (Training & Evaluation Loop)
for name, model in models.items():
    # التدريب
    model.fit(x_train, y_train)
    
    # التوقع على بيانات الاختبار
    y_test_pred = model.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    results[name] = test_acc
    
    print(f"\n" + "="*60)
    print(f"📊 Model: {name} | Test Accuracy: {test_acc:.4f}")
    print("="*60)
    
    # التقرير التفصيلي (Precision, Recall, F1)
    print(classification_report(y_test, y_test_pred))
    
    # رسم الـ Confusion Matrix (Heatmap)
    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# 4. تحديد الموديل الأفضل وحفظه (Model Saving)
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, 'heart_disease_model.pkl')

print(f"\n🏆 الموديل الفائز هو: {best_model_name}")
print(f"✅ تم حفظ الموديل بنجاح في ملف: 'heart_disease_model.pkl'")

# 5. Bonus: تحليل أهمية الخصائص (للموديلات الشجرية فقط)
if best_model_name in ["Random Forest", "XGBoost"]:
    print("\n🔍 جاري تحليل أهم العوامل (Feature Importance)...")
    importances = pd.Series(best_model.feature_importances_, index=x_train.columns)
    plt.figure(figsize=(8, 5))
    importances.nlargest(10).plot(kind='barh', color='teal')
    plt.title(f"Top 10 Important Features - {best_model_name}")
    plt.show()

    # 6. Bonus: SHAP Explainability (شرح التوقع)
    print("\n🚀 جاري إنشاء رسمة SHAP للـ Bonus...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test)