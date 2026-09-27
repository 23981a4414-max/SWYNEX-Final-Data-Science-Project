import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix,
                              mean_absolute_error, ConfusionMatrixDisplay)

df = pd.read_csv('water_breaks_clean.csv', parse_dates=['date'])

# =========================================================
# TASK A — Classification: is this a "legacy" (high-risk) pipe?
# Target only defined where material is known; we then use the trained
# model to *flag risk* for the 271 breaks with unknown material, and
# more importantly, to show which conditions predict legacy-pipe failures
# =========================================================
data = df.dropna(subset=['is_legacy_material']).copy()
data['is_legacy_material'] = data['is_legacy_material'].astype(int)

# zip: keep top few zip codes, bucket rest as 'Other'/'Unknown'
data['zip'] = data['zip'].fillna('Unknown').astype(str)
top_zips = data['zip'].value_counts().nlargest(6).index
data['zip_bucket'] = data['zip'].where(data['zip'].isin(top_zips), 'Other/Unknown')

features_cat = ['season', 'zip_bucket', 'break_category']
features_num = ['month', 'year']
X = data[features_cat + features_num]
y = data['is_legacy_material']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

pre = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat),
], remainder='passthrough')

clf = Pipeline([
    ('pre', pre),
    ('model', RandomForestClassifier(n_estimators=300, max_depth=5,
                                      min_samples_leaf=10, random_state=42))
])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_proba)
print("=== TASK A: Legacy-pipe risk classifier ===")
print(f"Test AUC: {auc:.3f}")
print(f"Baseline (majority class) accuracy: {max(y_test.mean(), 1-y_test.mean()):.3f}")
print(classification_report(y_test, y_pred, target_names=['Modern','Legacy']))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
print(f"5-fold CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(5,5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Modern','Legacy'],
                                          cmap='Blues', ax=ax, colorbar=False)
ax.set_title('Legacy-Pipe Classifier — Confusion Matrix')
plt.tight_layout()
plt.savefig('figs/06_confusion_matrix.png')
plt.close()

# Feature importance
ohe = clf.named_steps['pre'].named_transformers_['cat']
cat_names = list(ohe.get_feature_names_out(features_cat))
all_names = cat_names + features_num
importances = clf.named_steps['model'].feature_importances_
imp_df = pd.DataFrame({'feature': all_names, 'importance': importances}).sort_values('importance', ascending=True).tail(12)
fig, ax = plt.subplots(figsize=(8,6))
ax.barh(imp_df['feature'], imp_df['importance'], color='#4c72b0')
ax.set_title('Top Predictors of "Legacy Pipe" Break (Random Forest)')
ax.set_xlabel('Feature importance')
plt.tight_layout()
plt.savefig('figs/07_feature_importance.png')
plt.close()

# =========================================================
# TASK B — Forecasting: monthly break-count regression (maintenance load)
# =========================================================
monthly = df.set_index('date').resample('MS').size().rename('breaks').to_frame()
monthly = monthly[(monthly.index >= '2013-01-01') & (monthly.index <= '2026-07-01')]  # drop partial trailing month
monthly['month'] = monthly.index.month
monthly['year'] = monthly.index.year
monthly['freeze'] = monthly['month'].isin([11,12,1,2,3]).astype(int)
monthly['t'] = np.arange(len(monthly))  # time trend

X_m = monthly[['t','month','freeze']]
y_m = monthly['breaks']

# time-based split: train on all but last 12 months, test on last 12
split = len(monthly) - 12
Xtr, Xte = X_m.iloc[:split], X_m.iloc[split:]
ytr, yte = y_m.iloc[:split], y_m.iloc[split:]

pre_m = ColumnTransformer([
    ('month_oh', OneHotEncoder(handle_unknown='ignore'), ['month'])
], remainder='passthrough')
poisson_pipe = Pipeline([
    ('pre', pre_m),
    ('model', PoissonRegressor(alpha=0.5, max_iter=1000))
])
poisson_pipe.fit(Xtr, ytr)
pred = poisson_pipe.predict(Xte)
mae = mean_absolute_error(yte, pred)
naive_mae = mean_absolute_error(yte, [ytr.mean()]*len(yte))
print()
print("=== TASK B: Monthly break-volume forecast (Poisson regression) ===")
print(f"Test MAE (last 12 months): {mae:.2f} breaks/month")
print(f"Naive baseline (historical mean) MAE: {naive_mae:.2f} breaks/month")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(monthly.index, monthly['breaks'], label='Actual', color='#4c72b0', marker='o', ms=3)
ax.plot(Xte.index if hasattr(Xte,'index') else monthly.index[split:], pred,
        label='Forecast (last 12 mo, held out)', color='#c44e52', marker='o', ms=3)
ax.axvline(monthly.index[split], color='gray', linestyle='--', alpha=0.6)
ax.set_title('Monthly Break Volume: Actual vs Forecast')
ax.set_ylabel('Breaks per month')
ax.legend()
plt.tight_layout()
plt.savefig('figs/08_forecast.png')
plt.close()

print()
print("All modeling artifacts saved to figs/")
