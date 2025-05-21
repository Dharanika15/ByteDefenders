# ByteDefenders
A machine learning system that detects fraudulent credit card transactions in real-time using Decision Trees, Random Forest, and Logistic Regression. Designed for financial security, this tool analyzes transaction patterns to flag potential fraud with 92%+ accuracy
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, precision_recall_curve, 
                             average_precision_score, classification_report)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# 1. Data Loading Function
def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    
    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    display(df.head())
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Class Distribution ===")
    class_dist = df['is_fraud'].value_counts(normalize=True)
    print(class_dist)
    
    # Plot class distribution
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=class_dist.index, y=class_dist.values)
    plt.title('Class Distribution (0: Legitimate, 1: Fraud)')
    plt.ylabel('Percentage')
    plt.xlabel('Class')
    ax.set_xticklabels(['Legitimate', 'Fraud'])

    for p in ax.patches:
        ax.annotate(f'{p.get_height()*100:.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 10), textcoords='offset points')
    plt.savefig('class_distribution.png', bbox_inches='tight')
    plt.show()
    
    return df

# 2. Preprocessing Function
def preprocess_data(df):
    print("\nPreprocessing data...")
    
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
    df['transaction_dow'] = df['trans_date_trans_time'].dt.dayofweek
    
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    
    df['distance'] = np.sqrt(
        (df['lat'] - df['merch_lat'])**2 + 
        (df['long'] - df['merch_long'])**2
    )
    
    cols_to_drop = [
        'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last',
        'street', 'city', 'state', 'zip', 'lat', 'long', 'merch_lat',
        'merch_long', 'dob', 'job', 'unix_time', 'trans_num'
    ]
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    return df

# 3. Model Training and Evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names=None):
    print("\nTraining and evaluating models...")
    
    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8, min_samples_split=100, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=50, 
            class_weight='balanced', random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        avg_precision = average_precision_score(y_test, y_proba) if y_proba is not None else None

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc,
            'Avg Precision': avg_precision
        })

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legit', 'Fraud'], 
                    yticklabels=['Legit', 'Fraud'])
        plt.title(f'{name} - Confusion Matrix')
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png', bbox_inches='tight')
        plt.show()

        # ROC and PR Curve
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(f'roc_curve_{name.lower().replace(" ", "_")}.png', bbox_inches='tight')
            plt.show()

            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
            plt.figure(figsize=(6, 4))
            plt.plot(recall_curve, precision_curve, label=f'{name} (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.savefig(f'pr_curve_{name.lower().replace(" ", "_")}.png', bbox_inches='tight')
            plt.show()

        # Feature importance (only if we have feature names and the model supports it)
        if hasattr(model, 'feature_importances_') and feature_names is not None:
            plt.figure(figsize=(10, 6))
            feat_importances = pd.Series(model.feature_importances_, index=feature_names)
            feat_importances.nlargest(10).plot(kind='barh')
            plt.title(f'{name} - Feature Importance')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{name.lower().replace(" ", "_")}.png', bbox_inches='tight')
            plt.show()

    results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)

    # Model comparison bar chart
    plt.figure(figsize=(10, 6))
    results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar')
    plt.title('Model Comparison - Key Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison.png', bbox_inches='tight')
    plt.show()
    
    return results_df

# Main execution
if __name__ == "__main__":
    # 1. Load data
    filepath = r"C:\Users\Dharanika\Documents\fraudTest.csv"
    df = load_data(filepath)

    # 2. Preprocess data
    processed_df = preprocess_data(df)

    # 3. Prepare for modeling
    X = processed_df.drop('is_fraud', axis=1)
    y = processed_df['is_fraud']
    
    categorical_cols = ['category', 'gender']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
    
    # Apply transformations and get feature names
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after transformation
    numeric_features = numerical_cols
    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([numeric_features, categorical_features])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Handle imbalance
    print("\nApplying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # 5. Train and evaluate (pass feature names)
    results_df = train_and_evaluate_models(X_res, X_test, y_res, y_test, feature_names=feature_names)

    # 6. Save best model and preprocessor
    best_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=50, 
        class_weight='balanced', random_state=42, n_jobs=-1)
    best_model.fit(X_res, y_res)

    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')

    # 7. Save results
    print("\n=== Final Model Comparison ===")
    print(results_df)
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print("\nAll outputs saved successfully!")
