import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

# Φόρτωση δεδομένων
file_path = "Meta_Data.csv"
df = pd.read_csv(file_path, encoding='UTF-8')

print(f"Αρχικό σχήμα dataset: {df.shape}")

# Καθαρισμός: αφαίρεση στηλών με πολλά missing values
# Επέλεξα 40% ως threshold μετά από δοκιμές
threshold = 0.4 * len(df)
df = df.dropna(thresh=threshold, axis=1)
print(f"Μετά την αφαίρεση στηλών: {df.shape}")

df.fillna(0, inplace=True)

# Μετατροπή αριθμητικών στηλών που φορτώθηκαν ως strings
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = df[col].str.replace(',', '.').astype(float)
        except:
            pass

# Κανονικοποίηση ονομάτων στηλών
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

cleaned_file_path = "cleaned_meta_ads_data.xlsx"
df.to_excel(cleaned_file_path, index=False, engine='openpyxl')
print(f"Αποθηκεύτηκε: {cleaned_file_path}")

# Συνάρτηση ταξινόμησης καμπανιών
def classify_campaign_improved(name):
    """
    Ταξινομεί τις καμπάνιες βάσει του ονόματός τους σε κατηγορίες.
    Η σειρά των if statements είναι σημαντική για να αποφευχθούν overlaps.
    """
    name_lower = str(name).lower()
    name_original = str(name)
    
    if 'remarketing' in name_lower or 'retargeting' in name_lower:
        return 'Remarketing'
    
    if '[prospect]' in name_lower or 'prospect' in name_lower:
        return 'Prospecting'
    
    if 'traffic' in name_lower or 'view content' in name_lower:
        return 'Traffic'
    
    if 'engagement' in name_lower or 'contest' in name_lower:
        return 'Engagement'
    
    if any(word in name_lower for word in ['conversion', 'purchase', 'advantage+', 'abandon cart']):
        return 'Conversions'
    
    # Seasonal campaigns πρέπει να ελέγχονται πριν το Brand Awareness
    # γιατί μπορεί να περιέχουν "promo"
    if any(word in name_lower for word in ['sales', 'bazaar', 'black friday', 'black november', 
                                             'extended', 'christmas', 'summer', 'winter sales']):
        return 'Seasonal Sales'
    
    if any(word in name_lower for word in ['awareness', 'promo', 'teaser', 'coming soon', 
                                             'hello magazine', 'collection', 'local store',
                                             'motion video', 'graphic video', 'video campaign']):
        return 'Brand Awareness'
    
    # Catalog: περιλαμβάνει carousel, dynamic, broad και product-specific campaigns
    if any(word in name_lower for word in ['catalog', 'catalogue', 'carousel', 'sneaker', 
                                             'accessories', 'accesories', 'shoes', 'woman', 'man',
                                             'men', 'woman', 'dynamic', 'broad']):
        return 'Catalog Sales'
    
    if name_original.startswith('FB |') or 'old' in name_lower:
        return 'Catalog Sales'
    
    if 'iyc' in name_lower or '- iyc' in name_lower:
        if any(word in name_lower for word in ['ugc', 'boxer', 'video', 'image']):
            return 'Brand Awareness'
        return 'Catalog Sales'
    
    if 'new' in name_lower:
        return 'Brand Awareness'
    
    # Fallback - βάζω στο Catalog Sales γιατί είναι η πιο γενική κατηγορία
    return 'Catalog Sales'

df['Τύπος_Καμπάνιας'] = df['Όνομα_εκστρατείας'].apply(classify_campaign_improved)

print("\nΚατανομή τύπων καμπανιών:")
print(df['Τύπος_Καμπάνιας'].value_counts())

# Ορισμός επιτυχίας ανά τύπο καμπάνιας
# Τα thresholds βασίζονται στα descriptive statistics των δεδομένων
def classify_success_data_driven(row):
    type_ = row['Τύπος_Καμπάνιας']
    
    # Υπολογισμός ROAS
    roas = 0
    if row.get('Αξία_μετατροπών_από_αγορές', 0) > 0 and row.get('Έξοδα_EUR', 0) > 0:
        roas = row['Αξία_μετατροπών_από_αγορές'] / row['Έξοδα_EUR']
    
    cpa = float('inf')
    if row.get('Αγορές', 0) > 0 and row.get('Έξοδα_EUR', 0) > 0:
        cpa = row['Έξοδα_EUR'] / row['Αγορές']
    
    eng_rate = row['Αλληλεπίδραση_με_δημοσίευση'] / row['Απήχηση'] if row['Απήχηση'] > 0 else 0
    
    # Ορισμοί επιτυχίας - δοκίμασα διαφορετικά thresholds
    if type_ == 'Conversions':
        return int((roas >= 2.0) or (row['Αγορές'] >= 3 and cpa <= 25))
    
    elif type_ == 'Remarketing':
        return int((row['CTR_όλα'] >= 2.0) or (row['Αγορές'] >= 2))
    
    elif type_ == 'Prospecting':
        return int((row['Απήχηση'] >= 3000 and row['CPM_κόστος_ανά_1.000_εμφανίσεις_EUR'] <= 3.0) 
                   or (row['CTR_όλα'] >= 1.5))
    
    elif type_ == 'Catalog Sales':
        return int((row['Αγορές'] >= 2) or 
                   (row['CTR_όλα'] >= 1.5 and row['Προβολές_σελίδας_προορισμού'] >= 100))
    
    elif type_ == 'Traffic':
        return int((row['Κόστος_ανά_κλικ_όλα_EUR'] <= 0.5 and row['Κόστος_ανά_κλικ_όλα_EUR'] > 0) 
                   and (row['Προβολές_σελίδας_προορισμού'] >= 100 or row['Κλικ_σε_συνδέσμους'] >= 100))
    
    elif type_ == 'Engagement':
        return int((eng_rate >= 0.02) or (row['Αλληλεπίδραση_με_δημοσίευση'] >= 80))
    
    elif type_ == 'Brand Awareness':
        return int((row['CPM_κόστος_ανά_1.000_εμφανίσεις_EUR'] <= 2.5) 
                   and (row['Απήχηση'] >= 4000 or row['Συχνότητα'] >= 1.5))
    
    elif type_ == 'Seasonal Sales':
        return int((row['Αγορές'] >= 3) or (row['CTR_όλα'] >= 2.0))
    
    else:
        return int((row['CTR_όλα'] >= 1.5) or (eng_rate >= 0.02))

df['Επιτυχία'] = df.apply(classify_success_data_driven, axis=1)

print(f"\nΕπιτυχείς: {df['Επιτυχία'].sum()} ({df['Επιτυχία'].mean()*100:.1f}%)")
print(f"Ανεπιτυχείς: {(1-df['Επιτυχία']).sum()} ({(1-df['Επιτυχία'].mean())*100:.1f}%)")

success_by_type = df.groupby('Τύπος_Καμπάνιας')['Επιτυχία'].agg(['sum', 'count', 'mean'])
success_by_type.columns = ['Επιτυχείς', 'Σύνολο', 'Ποσοστό']
success_by_type['Ποσοστό'] = (success_by_type['Ποσοστό'] * 100).round(1)
print("\nΕπιτυχία ανά τύπο καμπάνιας:")
print(success_by_type.sort_values('Ποσοστό', ascending=False))

# Περιγραφικά στατιστικά
key_metrics = ['CTR_όλα', 'CPM_κόστος_ανά_1.000_εμφανίσεις_EUR', 'Κόστος_ανά_κλικ_όλα_EUR', 
               'Αγορές', 'Αλληλεπίδραση_με_δημοσίευση', 'Έξοδα_EUR', 'Απήχηση']

descriptive_stats = df.groupby('Τύπος_Καμπάνιας')[key_metrics].agg(['mean', 'median', 'std']).round(2)
top_types = df['Τύπος_Καμπάνιας'].value_counts().head(5).index
print("\nΠεριγραφικά στατιστικά (top 5 τύποι):")
print(descriptive_stats.loc[top_types])

# Μοντέλο πρόβλεψης - χρησιμοποιώ early indicators
print("\n--- Μοντέλο Πρόβλεψης ---")

# Features που είναι διαθέσιμα νωρίς στην καμπάνια
early_features = ['CTR_όλα', 'CPM_κόστος_ανά_1.000_εμφανίσεις_EUR', 'Κόστος_ανά_κλικ_όλα_EUR',
                  'Απήχηση', 'Εμφανίσεις', 'Συχνότητα', 'Τύπος_Καμπάνιας']

X = df[early_features].copy()
y = df['Επιτυχία']

# Split: 75% train, 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows")

# Pipeline για preprocessing
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = ['Τύπος_Καμπάνιας']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

# Grid Search για hyperparameter tuning
# Δοκίμασα max_depth 30 αλλά έκανε overfit
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 15, 20],
    'classifier__min_samples_split': [5, 10],
    'classifier__class_weight': ['balanced']
}

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

print("Εκτέλεση Grid Search...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

print(f"Βέλτιστες παράμετροι: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_

# Cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
print(f"Cross-validation F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nTest Set Results:")
print(classification_report(y_test, y_pred, target_names=['Ανεπιτυχής', 'Επιτυχής']))

if len(np.unique(y_test)) > 1:
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC Score: {auc_score:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ανεπιτυχής', 'Επιτυχής'],
            yticklabels=['Ανεπιτυχής', 'Επιτυχής'])
plt.title('Confusion Matrix')
plt.ylabel('Πραγματική Κλάση')
plt.xlabel('Προβλεπόμενη Κλάση')
plt.tight_layout()
plt.show()

# Feature importance
rf_model = best_model.named_steps['classifier']
ohe = best_model.named_steps['preprocessor'].named_transformers_['cat']
ohe_features = ohe.get_feature_names_out(['Τύπος_Καμπάνιας'])
ohe_features = [f.replace("Τύπος_Καμπάνιας_", "") for f in ohe_features]

all_features = numeric_features + list(ohe_features)
importances = rf_model.feature_importances_

feature_imp_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 15 σημαντικότερα χαρακτηριστικά:")
print(feature_imp_df.head(15).to_string(index=False))

plt.figure(figsize=(12, 8))
top_features = feature_imp_df.head(15)
plt.barh(top_features['Feature'], top_features['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Συσχετίσεις
corr_vars = ['CTR_όλα', 'CPM_κόστος_ανά_1.000_εμφανίσεις_EUR', 'Κόστος_ανά_κλικ_όλα_EUR',
             'Αγορές', 'Αλληλεπίδραση_με_δημοσίευση', 'Απήχηση', 'Εμφανίσεις', 
             'Έξοδα_EUR', 'Κλικ_σε_συνδέσμους']

corr_matrix = df[corr_vars].corr(method='pearson').round(2)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title("Πίνακας Συσχετίσεων")
plt.tight_layout()
plt.show()

# Συσχετίσεις ανά τύπο καμπάνιας
top_campaign_types = df['Τύπος_Καμπάνιας'].value_counts().head(3).index

for campaign_type in top_campaign_types:
    campaign_data = df[df['Τύπος_Καμπάνιας'] == campaign_type]
    if len(campaign_data) > 30:
        corr_camp = campaign_data[corr_vars].corr(method='pearson').round(2)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_camp, annot=True, cmap='coolwarm', center=0, fmt=".2f", 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title(f"Συσχετίσεις - {campaign_type}")
        plt.tight_layout()
        plt.show()

# Visualizations
plt.figure(figsize=(14, 6))
top_5_types = df['Τύπος_Καμπάνιας'].value_counts().head(5).index
data_top5 = df[df['Τύπος_Καμπάνιας'].isin(top_5_types)]
sns.boxplot(data=data_top5, x='Τύπος_Καμπάνιας', y='CTR_όλα')
plt.title('Κατανομή CTR ανά Τύπο Καμπάνιας')
plt.xticks(rotation=45, ha='right')
plt.ylabel('CTR (%)')
plt.tight_layout()
plt.show()

# Scatter plot CTR vs Purchases
plt.figure(figsize=(10, 6))
filtered_df = df[(df['CTR_όλα'] < 20) & (df['Αγορές'] < 30)]
sns.scatterplot(data=filtered_df, x='CTR_όλα', y='Αγορές', 
                hue='Επιτυχία', palette={0: 'red', 1: 'green'}, alpha=0.6, s=50)
plt.title('Σχέση CTR και Αγορών')
plt.xlabel('CTR (%)')
plt.ylabel('Αγορές')
plt.legend(title='Επιτυχία', labels=['Ανεπιτυχής', 'Επιτυχής'])
plt.tight_layout()
plt.show()

# Benchmarks για επιτυχημένες καμπάνιες
successful_campaigns = df[df['Επιτυχία'] == 1]

benchmarks = {
    'CTR (%)': successful_campaigns['CTR_όλα'].median(),
    'CPM (EUR)': successful_campaigns['CPM_κόστος_ανά_1.000_εμφανίσεις_EUR'].median(),
    'CPC (EUR)': successful_campaigns[successful_campaigns['Κόστος_ανά_κλικ_όλα_EUR'] > 0]['Κόστος_ανά_κλικ_όλα_EUR'].median(),
    'Απήχηση': successful_campaigns['Απήχηση'].median(),
    'Αγορές': successful_campaigns[successful_campaigns['Αγορές'] > 0]['Αγορές'].median()
}

print("\nMedian metrics για επιτυχημένες καμπάνιες:")
for metric, value in benchmarks.items():
    print(f"{metric}: {value:.2f}")

print("\nBenchmarks ανά τύπο καμπάνιας:")
top_5_types_filtered = [t for t in top_5_types if t != 'Other'][:5]
for camp_type in top_5_types_filtered:
    successful_type = df[(df['Επιτυχία'] == 1) & (df['Τύπος_Καμπάνιας'] == camp_type)]
    if len(successful_type) > 10:
        print(f"\n{camp_type}:")
        print(f"  Median CTR: {successful_type['CTR_όλα'].median():.2f}%")
        print(f"  Median CPM: {successful_type['CPM_κόστος_ανά_1.000_εμφανίσεις_EUR'].median():.2f} EUR")
        if successful_type[successful_type['Αγορές'] > 0]['Αγορές'].count() > 0:
            print(f"  Median Purchases: {successful_type[successful_type['Αγορές'] > 0]['Αγορές'].median():.1f}")

# Συνοψη
top_3_features = feature_imp_df.head(3)
print("\nTop 3 σημαντικότερα χαρακτηριστικά:")
for idx, row in top_3_features.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.3f}")

success_by_type_filtered = success_by_type[success_by_type.index != 'Other']
if len(success_by_type_filtered) > 0:
    best_type = success_by_type_filtered.sort_values('Ποσοστό', ascending=False).iloc[0]
    print(f"\nΚαλύτερος τύπος καμπάνιας: {best_type.name}")
    print(f"Success Rate: {best_type['Ποσοστό']:.1f}% ({int(best_type['Επιτυχείς'])}/{int(best_type['Σύνολο'])} campaigns)")

print(f"\nModel Performance:")
print(f"  F1-Score: {cv_scores.mean():.3f}")
print(f"  AUC-ROC: {auc_score:.3f}")

print("\nΠροτάσεις για νέες καμπάνιες:")
print(f"  Target CTR: >= {benchmarks['CTR (%)']:.1f}%")
print(f"  Target CPM: <= {benchmarks['CPM (EUR)']:.2f} EUR")
print(f"  Target CPC: <= {benchmarks['CPC (EUR)']:.3f} EUR")
