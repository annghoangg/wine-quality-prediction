"""
Script tạo visual cho Feature Engineering & Feature Selection
Dùng cho báo cáo thuyết trình về Wine Quality Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# Cấu hình chung
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'feature_engineering_visuals')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Màu sắc đẹp cho biểu đồ
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'dark': '#3B1F2B',
    'palette': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#44BBA4', '#E94F37', '#393E41'],
    'quality_palette': 'viridis',
}

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# Load và chuẩn bị dữ liệu
# ============================================================
print("=" * 60)
print("ĐANG TẢI DỮ LIỆU...")
print("=" * 60)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'wine_quality_dataset.csv')
df = pd.read_csv(DATA_PATH)

# Xử lý missing values
num_cols_with_na = df.columns[df.isnull().any()].tolist()
for col in num_cols_with_na:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# Tạo bản sao cho feature engineering
df_original = df.copy()
df_original['type'] = df_original['type'].map({'white': 0, 'red': 1})


# ============================================================
# Hàm Feature Engineering (giống trong notebook)
# ============================================================
def create_safe_wine_features(df_input):
    df_engineered = df_input.copy()
    df_engineered['acid ratio'] = df_engineered['fixed acidity'] / (df_engineered['volatile acidity'] + 1e-8)
    df_engineered['sulfur ratio'] = df_engineered['free sulfur dioxide'] / (df_engineered['total sulfur dioxide'] + 1e-8)
    df_engineered['alcohol sugar ratio'] = df_engineered['alcohol'] / (df_engineered['residual sugar'] + 1e-8)
    df_engineered['density alcohol interaction'] = df_engineered['density'] * df_engineered['alcohol']
    df_engineered['sulphates alcohol'] = df_engineered['sulphates'] * df_engineered['alcohol']
    df_engineered['alcohol squared'] = df_engineered['alcohol'] ** 2
    df_engineered['volatile acidity squared'] = df_engineered['volatile acidity'] ** 2
    df_engineered['log volatile acidity'] = np.log1p(df_engineered['volatile acidity'])
    df_engineered['log residual sugar'] = np.log1p(df_engineered['residual sugar'])
    df_engineered['log chlorides'] = np.log1p(df_engineered['chlorides'])
    df_engineered['quality score'] = (
        df_engineered['alcohol'] * 0.35 +
        (1 / (df_engineered['volatile acidity'] + 1e-8)) * 0.20 +
        df_engineered['sulphates'] * 0.20 +
        df_engineered['citric acid'] * 0.15 +
        (1 / (df_engineered['chlorides'] + 1e-8)) * 0.10
    )
    if 'type' in df_engineered.columns:
        df_engineered['type'] = df_engineered['type'].map({'white': 0, 'red': 1})
    return df_engineered


df_engineered = create_safe_wine_features(df)

# Danh sách 11 features mới
NEW_FEATURES = [
    'acid ratio',
    'sulfur ratio',
    'alcohol sugar ratio',
    'density alcohol interaction',
    'sulphates alcohol',
    'alcohol squared',
    'volatile acidity squared',
    'log volatile acidity',
    'log residual sugar',
    'log chlorides',
    'quality score'
]

# ============================================================
# VISUAL 1: So sánh Correlation trước và sau Feature Engineering
# ============================================================
print("\n[1/8] Tạo biểu đồ so sánh Correlation trước/sau FE...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Trước FE
corr_before = df_original.corrwith(df_original['quality']).drop('quality').abs().sort_values(ascending=False)
axes[0].barh(corr_before.index[:15], corr_before.values[:15], color=COLORS['primary'], edgecolor='white', linewidth=0.5)
axes[0].set_xlabel('|Correlation| với Quality', fontsize=12)
axes[0].set_title('TRƯỚC Feature Engineering\n(12 features gốc)', fontsize=14, fontweight='bold', color=COLORS['dark'])
axes[0].invert_yaxis()
for i, v in enumerate(corr_before.values[:15]):
    axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, color=COLORS['dark'])

# Sau FE
df_eng_numeric = df_engineered.copy()
df_eng_numeric['type'] = df_eng_numeric['type'].map({'white': 0, 'red': 1}) if df_eng_numeric['type'].dtype == 'object' else df_eng_numeric['type']
corr_after = df_eng_numeric.corrwith(df_eng_numeric['quality']).drop('quality').abs().sort_values(ascending=False)
colors_after = [COLORS['accent'] if feat in NEW_FEATURES else COLORS['primary'] for feat in corr_after.index[:15]]
axes[1].barh(corr_after.index[:15], corr_after.values[:15], color=colors_after, edgecolor='white', linewidth=0.5)
axes[1].set_xlabel('|Correlation| với Quality', fontsize=12)
axes[1].set_title('SAU Feature Engineering\n(23 features = 12 gốc + 11 mới)', fontsize=14, fontweight='bold', color=COLORS['dark'])
axes[1].invert_yaxis()
for i, v in enumerate(corr_after.values[:15]):
    axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, color=COLORS['dark'])

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS['primary'], label='Feature gốc'),
                   Patch(facecolor=COLORS['accent'], label='Feature mới (FE)')]
axes[1].legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.suptitle('SO SÁNH MỨC ĐỘ TƯƠNG QUAN VỚI QUALITY: TRƯỚC VÀ SAU FEATURE ENGINEERING', 
             fontsize=15, fontweight='bold', y=1.02, color=COLORS['dark'])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_correlation_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# VISUAL 2: Feature theo nhóm Quality cho RATIO features
# ============================================================
print("[2/8] Tạo biểu đồ phân bố Ratio Features theo Quality...")

ratio_features = ['acid ratio', 'sulfur ratio', 'alcohol sugar ratio']
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, feat in enumerate(ratio_features):
    quality_groups = df_engineered.groupby('quality')[feat].mean()
    bars = axes[idx].bar(quality_groups.index.astype(str), quality_groups.values, 
                         color=[plt.cm.viridis(i/len(quality_groups)) for i in range(len(quality_groups))],
                         edgecolor='white', linewidth=1)
    axes[idx].set_xlabel('Quality', fontsize=12)
    axes[idx].set_ylabel('Giá trị trung bình', fontsize=12)
    axes[idx].set_title(f'{feat.upper()}\ntheo từng mức Quality', fontsize=13, fontweight='bold')
    
    # Thêm giá trị lên đầu mỗi cột
    for bar, val in zip(bars, quality_groups.values):
        axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01*max(quality_groups.values),
                      f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.suptitle('RATIO FEATURES: GIÁ TRỊ TRUNG BÌNH THEO TỪNG MỨC CHẤT LƯỢNG RƯỢU', 
             fontsize=15, fontweight='bold', y=1.02, color=COLORS['dark'])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_ratio_features_by_quality.png'), dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# VISUAL 3: Interaction Features - Scatter plots
# ============================================================
print("[3/8] Tạo biểu đồ Interaction Features...")

interaction_features = ['density alcohol interaction', 'sulphates alcohol']
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for idx, feat in enumerate(interaction_features):
    scatter = axes[idx].scatter(df_engineered[feat], df_engineered['quality'], 
                                c=df_engineered['quality'], cmap='viridis', 
                                alpha=0.4, s=15, edgecolors='none')
    
    # Thêm đường trung bình theo quality
    quality_means = df_engineered.groupby('quality')[feat].mean()
    axes[idx].plot(quality_means.values, quality_means.index, 'r-o', linewidth=2.5, 
                   markersize=8, label='Trung bình', color=COLORS['success'], zorder=5)
    
    axes[idx].set_xlabel(f'{feat}', fontsize=12)
    axes[idx].set_ylabel('Quality', fontsize=12)
    axes[idx].set_title(f'{feat.upper()}\nvs Quality', fontsize=13, fontweight='bold')
    axes[idx].legend(fontsize=11)
    plt.colorbar(scatter, ax=axes[idx], label='Quality')

plt.suptitle('INTERACTION FEATURES: MỐI QUAN HỆ VỚI CHẤT LƯỢNG RƯỢU', 
             fontsize=15, fontweight='bold', y=1.02, color=COLORS['dark'])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_interaction_features.png'), dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# VISUAL 4: Polynomial Features - So sánh trước và sau biến đổi
# ============================================================
print("[4/8] Tạo biểu đồ Polynomial Features...")

poly_features = [
    ('alcohol', 'alcohol squared'),
    ('volatile acidity', 'volatile acidity squared')
]
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for row_idx, (orig_feat, poly_feat) in enumerate(poly_features):
    # BoxPlot gốc
    df_engineered.boxplot(column=orig_feat, by='quality', ax=axes[row_idx][0],
                          patch_artist=True,
                          boxprops=dict(facecolor=COLORS['primary'], alpha=0.7),
                          medianprops=dict(color=COLORS['success'], linewidth=2))
    axes[row_idx][0].set_title(f'{orig_feat.upper()}\n(Feature gốc)', fontsize=12, fontweight='bold')
    axes[row_idx][0].set_xlabel('Quality', fontsize=11)
    axes[row_idx][0].set_ylabel('Giá trị', fontsize=11)
    axes[row_idx][0].get_figure().suptitle('')

    # BoxPlot sau biến đổi
    df_engineered.boxplot(column=poly_feat, by='quality', ax=axes[row_idx][1],
                          patch_artist=True,
                          boxprops=dict(facecolor=COLORS['accent'], alpha=0.7),
                          medianprops=dict(color=COLORS['success'], linewidth=2))
    axes[row_idx][1].set_title(f'{poly_feat.upper()}\n(Sau biến đổi bình phương)', fontsize=12, fontweight='bold')
    axes[row_idx][1].set_xlabel('Quality', fontsize=11)
    axes[row_idx][1].set_ylabel('Giá trị', fontsize=11)
    axes[row_idx][1].get_figure().suptitle('')

plt.suptitle('POLYNOMIAL FEATURES: SO SÁNH TRƯỚC VÀ SAU BIẾN ĐỔI BÌNH PHƯƠNG', 
             fontsize=15, fontweight='bold', y=1.02, color=COLORS['dark'])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_polynomial_features.png'), dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# VISUAL 5: Logarithmic Features - So sánh phân phối trước/sau
# ============================================================
print("[5/8] Tạo biểu đồ Logarithmic Features...")

log_features = [
    ('volatile acidity', 'log volatile acidity'),
    ('residual sugar', 'log residual sugar'),
    ('chlorides', 'log chlorides')
]
fig, axes = plt.subplots(3, 2, figsize=(16, 16))

for row_idx, (orig_feat, log_feat) in enumerate(log_features):
    # Histogram gốc
    axes[row_idx][0].hist(df_engineered[orig_feat], bins=50, color=COLORS['primary'], 
                          alpha=0.7, edgecolor='white', linewidth=0.5)
    axes[row_idx][0].set_title(f'{orig_feat.upper()}\n(Phân phối gốc - SKEWED)', fontsize=12, fontweight='bold')
    axes[row_idx][0].set_xlabel('Giá trị', fontsize=11)
    axes[row_idx][0].set_ylabel('Tần suất', fontsize=11)
    skew_orig = df_engineered[orig_feat].skew()
    axes[row_idx][0].text(0.95, 0.95, f'Skewness = {skew_orig:.3f}', 
                          transform=axes[row_idx][0].transAxes,
                          ha='right', va='top', fontsize=11,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Histogram sau log
    axes[row_idx][1].hist(df_engineered[log_feat], bins=50, color=COLORS['accent'], 
                          alpha=0.7, edgecolor='white', linewidth=0.5)
    axes[row_idx][1].set_title(f'{log_feat.upper()}\n(Sau biến đổi Log)', fontsize=12, fontweight='bold')
    axes[row_idx][1].set_xlabel('Giá trị', fontsize=11)
    axes[row_idx][1].set_ylabel('Tần suất', fontsize=11)
    skew_log = df_engineered[log_feat].skew()
    axes[row_idx][1].text(0.95, 0.95, f'Skewness = {skew_log:.3f}', 
                          transform=axes[row_idx][1].transAxes,
                          ha='right', va='top', fontsize=11,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('LOGARITHMIC FEATURES: GIẢM ĐỘ LỆCH (SKEWNESS) CỦA PHÂN PHỐI', 
             fontsize=15, fontweight='bold', y=1.01, color=COLORS['dark'])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_log_features.png'), dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# VISUAL 6: Quality Score - Feature tổng hợp
# ============================================================
print("[6/8] Tạo biểu đồ Quality Score...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Box plot
df_engineered.boxplot(column='quality score', by='quality', ax=axes[0],
                      patch_artist=True,
                      boxprops=dict(facecolor=COLORS['secondary'], alpha=0.7),
                      medianprops=dict(color=COLORS['accent'], linewidth=2))
axes[0].set_title('QUALITY SCORE\ntheo từng mức Quality', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Quality', fontsize=12)
axes[0].set_ylabel('Quality Score', fontsize=12)
axes[0].get_figure().suptitle('')

# Violin plot
quality_vals = sorted(df_engineered['quality'].unique())
violin_data = [df_engineered[df_engineered['quality'] == q]['quality score'].values for q in quality_vals]
parts = axes[1].violinplot(violin_data, positions=range(len(quality_vals)), showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(plt.cm.viridis(i / len(quality_vals)))
    pc.set_alpha(0.7)
axes[1].set_xticks(range(len(quality_vals)))
axes[1].set_xticklabels([str(q) for q in quality_vals])
axes[1].set_xlabel('Quality', fontsize=12)
axes[1].set_ylabel('Quality Score', fontsize=12)
axes[1].set_title('QUALITY SCORE: PHÂN PHỐI\n(Violin Plot)', fontsize=13, fontweight='bold')

plt.suptitle('QUALITY SCORE: FEATURE TỔNG HỢP CÓ TRỌNG SỐ', 
             fontsize=15, fontweight='bold', y=1.02, color=COLORS['dark'])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_quality_score.png'), dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# VISUAL 7: Feature Importance - Đánh giá mức hữu ích của features mới
# ============================================================
print("[7/8] Tạo biểu đồ Feature Importance (đánh giá bằng mô hình)...")

# Chuẩn bị data
X = df_eng_numeric.drop('quality', axis=1)
y = df_eng_numeric['quality']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train a RF model on ALL features
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, 
                                   class_weight='balanced', n_jobs=-1)
rf_model.fit(X, y_encoded)

# Feature importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 8))
colors_imp = [COLORS['accent'] if feat in NEW_FEATURES else COLORS['primary'] for feat in importances.index]
bars = ax.barh(importances.index, importances.values, color=colors_imp, edgecolor='white', linewidth=0.5)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Random Forest)', fontsize=12)
ax.set_title('FEATURE IMPORTANCE: ĐÁNH GIÁ MỨC HỮU ÍCH CỦA FEATURES MỚI\n(Sử dụng Random Forest Classifier)', 
             fontsize=14, fontweight='bold', color=COLORS['dark'])

for bar, val in zip(bars, importances.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2., f'{val:.4f}', 
            va='center', fontsize=9)

legend_elements = [Patch(facecolor=COLORS['primary'], label='Feature gốc'),
                   Patch(facecolor=COLORS['accent'], label='Feature mới (FE)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_feature_importance.png'), dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# VISUAL 8: So sánh hiệu suất mô hình trước/sau Feature Engineering  
# ============================================================
print("[8/8] Tạo biểu đồ so sánh hiệu suất mô hình trước/sau FE...")

# Chuẩn bị data TRƯỚC FE
X_before = df_original.drop('quality', axis=1)
y_before = df_original['quality']
le_before = LabelEncoder()
y_before_enc = le_before.fit_transform(y_before)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_before, y_before_enc, test_size=0.2, random_state=42, stratify=y_before_enc)
scaler_b = StandardScaler()
X_train_b = scaler_b.fit_transform(X_train_b)
X_test_b = scaler_b.transform(X_test_b)
# Tính k_neighbors phù hợp cho SMOTE
from collections import Counter
min_count_b = min(Counter(y_train_b).values())
k_neighbors_b = min(5, min_count_b - 1) if min_count_b > 1 else 1
smote_b = SMOTE(random_state=42, k_neighbors=k_neighbors_b)
X_train_b_res, y_train_b_res = smote_b.fit_resample(X_train_b, y_train_b)

# Chuẩn bị data SAU FE (Top 15 features)
feature_correlation = X.corrwith(pd.Series(y_encoded, index=X.index)).abs().sort_values(ascending=False)
top_features = feature_correlation.head(15).index.tolist()
X_after = X[top_features]

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_after, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
scaler_a = StandardScaler()
X_train_a = scaler_a.fit_transform(X_train_a)
X_test_a = scaler_a.transform(X_test_a)
min_count_a = min(Counter(y_train_a).values())
k_neighbors_a = min(5, min_count_a - 1) if min_count_a > 1 else 1
smote_a = SMOTE(random_state=42, k_neighbors=k_neighbors_a)
X_train_a_res, y_train_a_res = smote_a.fit_resample(X_train_a, y_train_a)

# Các mô hình đánh giá
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced', n_jobs=-1),
    'Extra Trees': ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced', n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, objective='multi:softprob', random_state=42, n_jobs=-1),
}

results_before = {}
results_after = {}

for name, model in models.items():
    # Trước FE
    model_b = model.__class__(**model.get_params())
    model_b.fit(X_train_b_res, y_train_b_res)
    y_pred_b = model_b.predict(X_test_b)
    f1_b = f1_score(y_test_b, y_pred_b, average='macro')
    results_before[name] = f1_b
    
    # Sau FE
    model_a = model.__class__(**model.get_params())
    model_a.fit(X_train_a_res, y_train_a_res)
    y_pred_a = model_a.predict(X_test_a)
    f1_a = f1_score(y_test_a, y_pred_a, average='macro')
    results_after[name] = f1_a

# Vẽ biểu đồ so sánh
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(models))
width = 0.35

bars_before = ax.bar(x - width/2, list(results_before.values()), width, 
                     label='Trước FE (12 features gốc)', color=COLORS['primary'], edgecolor='white', linewidth=1)
bars_after = ax.bar(x + width/2, list(results_after.values()), width, 
                    label='Sau FE (Top 15 features)', color=COLORS['accent'], edgecolor='white', linewidth=1)

ax.set_ylabel('F1-Score (Macro)', fontsize=13)
ax.set_title('SO SÁNH HIỆU SUẤT MÔ HÌNH: TRƯỚC VÀ SAU FEATURE ENGINEERING\n(F1-Score Macro trên tập Test)', 
             fontsize=14, fontweight='bold', color=COLORS['dark'])
ax.set_xticks(x)
ax.set_xticklabels(list(models.keys()), fontsize=12)
ax.legend(fontsize=12)

# Thêm giá trị
for bar in bars_before:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.003, f'{height:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['primary'])
for bar in bars_after:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.003, f'{height:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['accent'])

# Thêm mũi tên chỉ sự cải thiện
for i, (name, f1_b) in enumerate(results_before.items()):
    f1_a = results_after[name]
    improvement = (f1_a - f1_b) / f1_b * 100
    color = '#2E7D32' if improvement > 0 else '#C62828'
    symbol = '↑' if improvement > 0 else '↓'
    ax.text(i, max(f1_b, f1_a) + 0.02, f'{symbol} {abs(improvement):.1f}%', 
            ha='center', fontsize=12, fontweight='bold', color=color)

ax.set_ylim(0, max(max(results_before.values()), max(results_after.values())) + 0.06)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_model_performance_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# In kết quả tổng hợp
# ============================================================
print("\n" + "=" * 60)
print("HOÀN THÀNH! Tất cả visual đã được lưu tại:")
print(f"  {OUTPUT_DIR}")
print("=" * 60)
print("\nDanh sách các file visual:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  📊 {f} ({size_kb:.0f} KB)")

print("\n\nKẾT QUẢ SO SÁNH HIỆU SUẤT MÔ HÌNH:")
print("-" * 50)
print(f"{'Mô hình':<20} {'Trước FE':>12} {'Sau FE':>12} {'Cải thiện':>12}")
print("-" * 50)
for name in models:
    f1_b = results_before[name]
    f1_a = results_after[name]
    improvement = (f1_a - f1_b) / f1_b * 100
    print(f"{name:<20} {f1_b:>12.4f} {f1_a:>12.4f} {improvement:>+11.1f}%")
print("-" * 50)
