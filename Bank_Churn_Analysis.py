# ---------------------------
# Bank Customer Churn Analysis
# Objective: Reduce customer churn to increase bank revenue
# ---------------------------
# Guide Questions
# 1. Which attributes are more common among churners than non-churners?
# 2. Can churn be predicted using the available variables?
# 3. What are the overall demographics of the bank's customers?
# 4. Are there differences in account behavior by geography?
# 5. What customer segments exist within the bank's customer base?

# ---------------------------
# 1. Load & Clean Data
# ---------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1A. Load data
# ---------------------------
file_path = "Bank_Churn_Messy.xlsx"
demographics = pd.read_excel(file_path, sheet_name='Customer_Info')
account = pd.read_excel(file_path, sheet_name='Account_Info')

# ---------------------------
# 1B. Clean Numeric Columns
# ---------------------------
demographics['EstimatedSalary'] = demographics['EstimatedSalary'].replace('[€,]', '', regex=True).astype(float)
account['Balance'] = account['Balance'].replace('[€,]', '', regex=True).astype(float)
account['HasCrCard'] = account['HasCrCard'].map({'Yes': 1, 'No': 0})
account['IsActiveMember'] = account['IsActiveMember'].map({'Yes': 1, 'No': 0})

# ---------------------------
# 1C. Drop Redundant Columns & Merge
# ---------------------------
account = account.drop(columns=['Tenure']).drop_duplicates(subset='CustomerId')
df = pd.merge(demographics, account, on='CustomerId', how='inner')
df = df[['CustomerId','Surname','CreditScore','Geography','Gender','Age','Tenure',
         'Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']]

# ---------------------------
# 1D. Geography Cleaning
# ---------------------------
country_mapping = {'FRA': 'France', 'French': 'France', 'France': 'France',
                   'Germany': 'Germany', 'Spain': 'Spain'}
df['Geography'] = df['Geography'].replace(country_mapping)

# ---------------------------
# 1E. Quality Checks
# ---------------------------
df = df.dropna(subset=['Surname', 'Age'])
print(df.head())
print(df.info())
print(df.isnull().sum())

# ---------------------------
# 2. Churn Analysis & Customer Profiles
# ---------------------------

# ---------------------------
# 2A. Overall Churn Rate
# ---------------------------
overall_churn = df['Exited'].mean()
print(f"Overall churn rate: {overall_churn:.2%}")

# ---------------------------
# 2B. Churn by Geography
# ---------------------------
geo_churn = df.groupby('Geography')['Exited'].mean().reset_index()
plt.figure(figsize=(8,6))
sns.barplot(x='Geography', y='Exited', data=geo_churn, palette="viridis")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
plt.ylim(0, 0.4)
plt.title("Churn Rate by Geography", fontsize=14)
plt.ylabel("Churn Rate (%)", fontsize=12)
plt.xlabel("Geography", fontsize=12)
for i, row in geo_churn.iterrows():
    plt.text(i, row['Exited'] + 0.01, f"{row['Exited']*100:.1f}%", ha='center', fontweight='bold')
plt.show()

## Observation: Germany ~32%, France & Spain ~16%
## Action: Target retention campaigns in Germany

# ---------------------------
# 2C. Churn by Gender
# ---------------------------
gender_churn = df.groupby('Gender')['Exited'].mean().reset_index()
plt.figure(figsize=(8,6))
sns.barplot(x='Gender', y='Exited', data=gender_churn, palette="viridis")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
plt.ylim(0, 0.30)
plt.title("Churn Rate by Gender", fontsize=14)
plt.ylabel("Churn Rate (%)", fontsize=12)
plt.xlabel("Gender", fontsize=12)
for i, row in gender_churn.iterrows():
    plt.text(i, row['Exited'] + 0.01, f"{row['Exited']*100:.1f}%", ha='center', fontweight='bold')
plt.show()

## Observation: Female churn higher (~25% vs 16.5%)
## Action: Investigate female customer behavior & engagement

# ---------------------------
# 2D. Churn by Age Group
# ---------------------------
age_bins = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 70],
                  labels=['18–30','31–40','41–50','51–60','61–70'])
age_churn = df.groupby(age_bins)['Exited'].mean().reset_index()
age_churn.columns = ['Age Group', 'Exited']
plt.figure(figsize=(8,6))
sns.barplot(x='Age Group', y='Exited', data=age_churn, palette="viridis")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
plt.ylim(0, 0.65)
plt.title("Churn Rate by Age Group", fontsize=14)
plt.ylabel("Churn Rate (%)", fontsize=12)
plt.xlabel("Age Group", fontsize=12)
for i, row in age_churn.iterrows():
    plt.text(i, row['Exited'] + 0.02, f"{row['Exited']*100:.1f}%", ha='center', fontweight='bold')
plt.show()

## Observation: 50–60 highest churn (~56%), 18–30 lowest (~7.5%)
## Action: Tailor retention for older customers

# ---------------------------
# 2E. Churn by Credit Score Group
# ---------------------------
credit_bins = [300, 579, 669, 739, 850]
credit_labels = ['Poor', 'Fair', 'Good', 'Excellent']
df['CreditScoreGroup'] = pd.cut(df['CreditScore'], bins=credit_bins, labels=credit_labels)
credit_churn = df.groupby('CreditScoreGroup')['Exited'].agg(['count','mean']).reset_index()
credit_churn.columns = ['CreditScoreGroup','CustomerCount','ChurnRate']
plt.figure(figsize=(8,6))
sns.barplot(x='CreditScoreGroup', y='ChurnRate', data=credit_churn, order=credit_labels, palette="viridis")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
plt.ylim(0.16, 0.23)
plt.title("Churn Rate by Credit Score Group", fontsize=14)
plt.ylabel("Churn Rate (%)", fontsize=12)
plt.xlabel("Credit Score Group", fontsize=12)
for i, row in credit_churn.iterrows():
    plt.text(i, row['ChurnRate'] + 0.005, f"{row['ChurnRate']*100:.1f}%", ha='center', fontweight='bold')
plt.show()

## Observation: Poor churn highest (~22%), Good lowest (~18.6%), Excellent slightly higher (~20%)
## Action: Implement credit-based retention strategies

# ---------------------------
# 3A. Predictive Modeling Setup
# ---------------------------
features = ['CreditScore','Geography','Gender','Age','Tenure','Balance',
            'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ---------------------------
# 3B. Random Forest
# ---------------------------
rf_model = RandomForestClassifier(n_estimators=300, min_samples_split=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

## Observation: 86.7% accuracy, recall 51%, better at identifying churners
## Action: Use Random Forest for retention prioritization

# ---------------------------
# 3C. Feature Importance (Random Forest)
# ---------------------------
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=rf_importance.head(10), x='Importance', y='Feature', palette="viridis")
plt.title("Top Drivers of Customer Churn", fontsize=14)
plt.xlabel("Relative Impact on Churn (%)", fontsize=12)
plt.ylabel("Customer Attribute", fontsize=12)
plt.tight_layout()
plt.show()

## Observation: Age, NumOfProducts, Balance, Salary are top predictors
## Action: Age-specific retention, cross-sell products, premium retention offers

# ---------------------------
# 4A. Customer Churn Risk Segmentation
# ---------------------------
results = X_test.copy()

results['Actual'] = y_test.values
results['Churn_Prob'] = rf_model.predict_proba(X_test)[:,1]
results['Prediction'] = rf_model.predict(X_test)
results['Risk_Tier'] = pd.cut(results['Churn_Prob'], bins=[0,0.40,0.70,1],
                              labels=['Low Risk','Medium Risk','High Risk'])

# ---------------------------
# 4B. Risk Tier Summary & Profiling
# ---------------------------
risk_counts = results['Risk_Tier'].value_counts().sort_index()
print("Customer Count per Risk Tier:")
print(risk_counts)

risk_stats = results.groupby('Risk_Tier')[['Age','Balance','NumOfProducts','EstimatedSalary']].mean().round(2)
print("\nAverage Customer Profile by Risk Tier:")
print(risk_stats)

# Observation:
# - High Risk (~7% of customers): older, more products, higher balances and salaries
# - Medium Risk (~10%): moderate profiles
# - Low Risk (~83%): younger, fewer products, lower balances
# Action: Prioritize retention campaigns on Medium & High Risk tiers using top predictors

# ---------------------------
# 4C. High-Risk Customer Exploration
# ---------------------------
high_risk_customers = results[results['Risk_Tier'] == 'High Risk']
print("\nTop 10 High-Risk Customers:")
print(high_risk_customers.head(10)[['Age','Balance','NumOfProducts','EstimatedSalary']])

# Observation:
# - High Risk customers tend to be older (45–65)
# - Often show high balances and diverse product usage patterns
# - These accounts should be prioritized for retention

# ---------------------------
# 5. Actionable Recommendations & Retention Strategies
# ---------------------------

# High-Risk Customers (~7% of customers)
# - Profile: Older customers (45–65), high balances, multiple products, higher salaries.
# - Recommended Actions:
#     - Personalized premium offers and loyalty perks
#     - High-touch outreach (calls, account managers, relationship managers)
#     - Cross-sell or upsell based on product usage patterns

# Medium-Risk Customers (~10% of customers)
# - Profile: Moderate age, balances, and product usage
# - Recommended Actions:
#     - Targeted emails or app notifications highlighting product benefits
#     - Promotions to increase engagement
#     - Incentives to adopt additional products

# Low-Risk Customers (~83% of customers)
# - Profile: Younger customers, fewer products, lower balances
# - Recommended Actions:
#     - Maintain engagement through newsletters, app notifications, and digital touchpoints
#     - Monitor for early churn indicators
#     - Encourage gradual adoption of additional products

# ---------------------------
# Factor-Specific Interventions (Top Drivers of Churn)
# ---------------------------

# Age: Focus retention efforts on the 50–60 age group
# Number of Products: Encourage adoption of additional products through incentives or bundled offers
# Balance: Monitor accounts with high balances for early churn signals and provide tailored retention offers
# Estimated Salary: Offer financial incentives or premium services to retain high-value customers
# Credit Score: Tailor retention offers for high-credit-score customers (e.g., exclusive benefits or loyalty programs)

# End of Analysis