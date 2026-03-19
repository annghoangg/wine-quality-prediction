import pandas as pd
import numpy as np
import warnings
import os
import joblib
import optuna

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

def create_safe_wine_features(df_input):
    df_engineered = df_input.copy()
    
    # 1. Ratio features
    df_engineered['acid ratio'] = df_engineered['fixed acidity'] / (df_engineered['volatile acidity'] + 1e-8)
    df_engineered['sulfur ratio'] = df_engineered['free sulfur dioxide'] / (df_engineered['total sulfur dioxide'] + 1e-8)
    df_engineered['alcohol sugar ratio'] = df_engineered['alcohol'] / (df_engineered['residual sugar'] + 1e-8)
    
    # 2. Interaction features
    df_engineered['density alcohol interaction'] = df_engineered['density'] * df_engineered['alcohol']
    df_engineered['sulphates alcohol'] = df_engineered['sulphates'] * df_engineered['alcohol']
    
    # 3. Polynomial features
    df_engineered['alcohol squared'] = df_engineered['alcohol'] ** 2
    df_engineered['volatile acidity squared'] = df_engineered['volatile acidity'] ** 2
    
    # 4. Logarithmic features
    df_engineered['log volatile acidity'] = np.log1p(df_engineered['volatile acidity'])
    df_engineered['log residual sugar'] = np.log1p(df_engineered['residual sugar'])
    df_engineered['log chlorides'] = np.log1p(df_engineered['chlorides'])
    
    # 5. Quality Score 
    df_engineered['chemical_balance_index'] = (
        df_engineered['alcohol'] * 0.35 +
        (1 / (df_engineered['volatile acidity'] + 1e-8)) * 0.20 +
        df_engineered['sulphates'] * 0.20 +
        df_engineered['citric acid'] * 0.15 +
        (1 / (df_engineered['chlorides'] + 1e-8)) * 0.10
    )
    
    # Encode 'type'
    if 'type' in df_engineered.columns:
        df_engineered['type'] = df_engineered['type'].map({'white': 0, 'red': 1})
        
    return df_engineered

print("Loading data...")
# Read dataset
df = pd.read_csv('wine_quality_dataset.csv')

# Handle missing values as original code did
num_cols_with_na = df.columns[df.isnull().any()].tolist()
for col in num_cols_with_na:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

df_engineered = create_safe_wine_features(df)

X = df_engineered.drop('quality', axis=1)
y = df_engineered['quality']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

feature_correlation = X.corrwith(y).abs().sort_values(ascending=False)
top_features = feature_correlation.head(15).index.tolist()
X_selected = X[top_features]

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

min_class_count = pd.Series(y_train).value_counts().min()
k_neighbors = min(3, min_class_count - 1) if min_class_count > 1 else 1

smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 10, 30, step=5),
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X_train_resampled, y_train_resampled, cv=3, scoring='f1_macro', n_jobs=-1).mean()

print("Training Random Forest...")
study_rf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_rf.optimize(optimize_rf, n_trials=5)
best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42, class_weight='balanced')
best_rf.fit(X_train_resampled, y_train_resampled)

print("Training Extra Trees...")
best_et = ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced', n_jobs=-1)
best_et.fit(X_train_resampled, y_train_resampled)

print("Training XGBoost...")
best_xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, 
                         objective='multi:softprob', random_state=42, n_jobs=-1)
best_xgb.fit(X_train_resampled, y_train_resampled)

print("Training CatBoost...")
cat_model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, 
                               random_seed=42, verbose=False, auto_class_weights='Balanced')
cat_model.fit(X_train_resampled, y_train_resampled)

print("Training Stacking Classifier...")
stacking_clf = StackingClassifier(
    estimators=[('rf', best_rf), ('et', best_et), ('xgb', best_xgb)],
    final_estimator=LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=500),
    cv=3, n_jobs=-1
)
stacking_clf.fit(X_train_resampled, y_train_resampled)

print("Training Blending Classifier...")
rf_train_proba = best_rf.predict_proba(X_train_resampled)
et_train_proba = best_et.predict_proba(X_train_resampled)
xgb_train_proba = best_xgb.predict_proba(X_train_resampled)
blend_X_train = np.hstack((rf_train_proba, et_train_proba, xgb_train_proba))
blend_model = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=500, random_state=42)
blend_model.fit(blend_X_train, y_train_resampled)

def evaluate_model(name, model, is_blend=False):
    if is_blend:
        rf_test_proba = best_rf.predict_proba(X_test_scaled)
        et_test_proba = best_et.predict_proba(X_test_scaled)
        xgb_test_proba = best_xgb.predict_proba(X_test_scaled)
        blend_X_test = np.hstack((rf_test_proba, et_test_proba, xgb_test_proba))
        preds_encoded = model.predict(blend_X_test)
    else:
        preds_encoded = model.predict(X_test_scaled)
        
    f1 = f1_score(y_test, preds_encoded, average='macro')
    return f1

results = {
    'Random Forest': evaluate_model('RF', best_rf),
    'Extra Trees': evaluate_model('ET', best_et),
    'XGBoost': evaluate_model('XGB', best_xgb),
    'CatBoost': evaluate_model('CAT', cat_model),
    'Stacking': evaluate_model('Stacking', stacking_clf),
    'Blending': evaluate_model('Blending', blend_model, is_blend=True)
}

print("\n=== FINAL TEST SET PERFORMANCE (F1 Macro) ===")
for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:15s} : {score:.4f}")

best_model_name = max(results, key=results.get)
print(f"\nBest Model: {best_model_name}")

if not os.path.exists('models'):
    os.makedirs('models')

print("Saving preprocessing objects...")
joblib.dump(imputer, 'models/imputer.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')
joblib.dump(top_features, 'models/top_features.joblib')

print("Saving models...")
# Always save the best base models in case they are needed for blending
joblib.dump(best_rf, 'models/best_rf.joblib')
joblib.dump(best_et, 'models/best_et.joblib')
joblib.dump(best_xgb, 'models/best_xgb.joblib')

if best_model_name == 'Random Forest':
    best_model_obj = best_rf
elif best_model_name == 'Extra Trees':
    best_model_obj = best_et
elif best_model_name == 'XGBoost':
    best_model_obj = best_xgb
elif best_model_name == 'CatBoost':
    best_model_obj = cat_model
elif best_model_name == 'Stacking':
    best_model_obj = stacking_clf
elif best_model_name == 'Blending':
    best_model_obj = blend_model

if best_model_name in ['Random Forest', 'Extra Trees', 'XGBoost', 'CatBoost', 'Stacking', 'Blending']:
    joblib.dump(best_model_obj, 'models/best_model.joblib')
    
# Write a tiny metadata file for app.py
with open('models/metadata.txt', 'w') as f:
    f.write(best_model_name)

print("Export complete! Files saved in 'models/' directory.")
