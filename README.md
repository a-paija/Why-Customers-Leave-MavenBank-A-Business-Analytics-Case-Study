# Customer Churn Analysis & Retention Strategy

## 📊 Business Case Scenario
Mavenland Bank is facing an unexpected rise in customer churn that is threatening quarterly revenue targets. The executive team from Product, Marketing, and Customer Success departments have convened an emergency meeting and need your analytical expertise to understand and address this critical issue.

**Business Objective:** Reduce customer churn by 25% within two quarters to stabilize quarterly revenue targets.

## 🛠️ Tech Stack

- Languages: Python (Pandas, NumPy)
- Data Visualization: Seaborn, Matplotlib
- Modeling: Scikit-Learn (Random Forest)
- Tools: Jupyter Notebook, GitHub
- Business Focus: Customer Segmentation, Retention Strategy, Churn Prediction

## 🎯 Key Questions Answered

1. Which customer attributes are more common among churners than non-churners?
2. Can churn be accurately predicted using the available customer and account variables?
3. How does customer behavior differ across geographic regions (France, Germany, Spain)?
4. How does churn vary across demographic groups such as age, gender, and credit score categories?
5. Which features are the strongest drivers of churn based on predictive modeling?
6. What actionable retention strategies can be derived from profiling high-risk customer segments?

## 📁 Dataset
- **Source:** Messy Excel file with two tables (`Customer_Info` and `Account_Info`)
- **Records:** 10,000 customers
- **Features:** 13 attributes including demographics, account behavior, and churn status

## 🧹 Data Cleaning Steps
- Standardized numeric formats (currencies → floats)  
- Cleaned inconsistent geography labels  
- Handled missing values & duplicates  
- Merged customer and account tables  
- Engineered features (age groups, product counts, etc.)

## 🔍 Key Findings

### Demographic Insights
- **Overall Churn Rate:** 20.4% (above industry benchmarks)
- **Geography:** Germany has 32% churn vs 16% for France/Spain
- **Gender:** Female customers churn at 25% vs 16.5% for males
- **Age:** Customers aged 50-60 have 56% churn rate vs 7.5% for 18-30 age group
- **Credit Score:** Poor credit customers (22% churn) vs Good credit (18.6% churn)

### Predictive Modeling
- **Model:** Random Forest Classifier
- **Accuracy:** 86.7%
- **Key Predictors:** Age, Number of Products, Balance, Estimated Salary

## 🎯 Customer Risk Segmentation

### Three-Tier Risk Classification
| Risk Tier | % of Customers | Profile Characteristics | Recommended Actions |
|-----------|----------------|------------------------|-------------------|
| **High Risk** | 7% | Older (45-65), high balances, multiple products | Premium retention offers, dedicated relationship managers |
| **Medium Risk** | 10% | Moderate age, balances, product usage | Targeted campaigns, engagement incentives |
| **Low Risk** | 83% | Younger, fewer products, lower balances | Maintain engagement, monitor for changes |
  
## 🔍 Exploratory Analysis & Code Snippets

<details>
<summary><strong>Churn Analysis & Customer Profiles</strong></summary> 

### Overall Churn Rate

```python
overall_churn = df['Exited'].mean()
print(f"Overall churn rate: {overall_churn:.2%}")
```
```
- Overall churn rate is 20.37%, indicating that roughly 1 in 5 customers leave the bank.
```

### Churn by Geography
  
```python
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
```
<img src="https://github.com/a-paija/Why-Customers-Leave-MavenBank-A-Business-Analytics-Case-Study/blob/main/Images/Churn%20Rate%20by%20Geography.png" alt="Churn Rate by Geography" width="500" height="600"/>

```
- Germany shows the highest churn (~32%), while France and Spain remain lower at ~16%.
```

### Churn by Gender
```python
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
```
<img src="https://github.com/a-paija/Why-Customers-Leave-MavenBank-A-Business-Analytics-Case-Study/blob/main/Images/Churn%20Rate%20by%20Gender.png" alt="Churn Rate by Gender" width="500" height="600"/>

```
- Female customers churn at ~25%, significantly higher than male customers at ~16.5%.
```

### Churn by Age Group

```python
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
```
<img src="https://github.com/a-paija/Why-Customers-Leave-MavenBank-A-Business-Analytics-Case-Study/blob/main/Images/Churn%20Rate%20by%20Age%20Group.png" alt="Churn Rate by Age Group" width="500" height="600"/>

```
- Customers aged 51–60 churn the most (~56%), while ages 18–30 churn the least (~7.5%).
```

### Churn by Credit Score Group

```python
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
```
<img src="https://github.com/a-paija/Why-Customers-Leave-MavenBank-A-Business-Analytics-Case-Study/blob/main/Images/Churn%20Rate%20by%20Credit%20Score%20Group.png" alt="Churn Rate by Credit Group" width="500" height="600"/>

```
- Customers with Poor credit show the highest churn (~22%).
- Those with Good scores have the lowest churn (~18.6%).
- Excellent credit users churn slightly more (~20%) than Good.
```

---

</details>

<details> 
<summary><strong>Predictive Modeling & Feature Insights</strong></summary>

```python
features = ['CreditScore','Geography','Gender','Age','Tenure','Balance',
            'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
``` 
```python
rf_model = RandomForestClassifier(n_estimators=300, min_samples_split=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
```
```
- Observation: 86.7% accuracy, recall 51%, better at identifying churners
```

```python
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
```
<img src="https://github.com/a-paija/Why-Customers-Leave-MavenBank-A-Business-Analytics-Case-Study/blob/main/Images/Top%20Drivers%20of%20Customer%20Churn.png" alt="Churn Rate by Credit Group" width="700" height="800"/>

```
- Observation: Age, NumOfProducts, Balance, Salary are top predictors
```
---

</details>

<details> 
<summary><strong>Customer Churn Risk Segmentation & High-Risk Profiling</strong></summary>
  
```python
results = X_test.copy()

results['Actual'] = y_test.values
results['Churn_Prob'] = rf_model.predict_proba(X_test)[:,1]
results['Prediction'] = rf_model.predict(X_test)
results['Risk_Tier'] = pd.cut(results['Churn_Prob'], bins=[0,0.40,0.70,1],
                              labels=['Low Risk','Medium Risk','High Risk'])
```
```
- Churn probabilities successfully segmented into Low, Medium, and High Risk tiers.
- Enables targeted retention strategies based on churn likelihood.
```
```python
risk_counts = results['Risk_Tier'].value_counts().sort_index()
print("Customer Count per Risk Tier:")
print(risk_counts)

risk_stats = results.groupby('Risk_Tier')[['Age','Balance','NumOfProducts','EstimatedSalary']].mean().round(2)
print("\nAverage Customer Profile by Risk Tier:")
print(risk_stats)
```

```
- Low Risk: 2,473 customers
- Medium Risk: 292 customers
- High Risk: 231 customers
```
### Average Customer Profile by Risk Tier

```
| **Risk Tier**   | **Age** | **Balance**   | **NumOfProducts** | **EstimatedSalary** |
|-----------------|---------|----------------|--------------------|----------------------|
| **Low Risk**    | 36.91   | 71,747.72      | 1.53               | 98,995.27            |
| **Medium Risk** | 45.86   | 96,124.23      | 1.24               | 97,689.29            |
| **High Risk**   | 51.02   | 92,645.82      | 1.78               | 104,118.17           |
```

### Insights
```
- High Risk customers are older (51+), have higher balances, and more product holdings.
- Medium Risk customers have moderate age but elevated balances.
- Low Risk customers skew younger with fewer products and lower balances.
```

```python
high_risk_customers = results[results['Risk_Tier'] == 'High Risk']
print("\nTop 10 High-Risk Customers:")
print(high_risk_customers.head(10)[['Age','Balance','NumOfProducts','EstimatedSalary']])
```
```
| **Age** | **Balance**    | **NumOfProducts** | **EstimatedSalary** |
|--------:|---------------:|-------------------:|---------------------:|
| 53      | 156,674.20     | 1                 | 118,502.34           |
| 48      | 118,317.27     | 4                 | 78,702.98            |
| 63      | 110,314.21     | 2                 | 37,464.00            |
| 56      | 143,249.67     | 1                 | 88,428.41            |
| 45      | 103,583.05     | 1                 | 132,127.69           |
| 50      | 112,650.89     | 1                 | 166,386.22           |
| 65      | 120,100.41     | 1                 | 107,563.16           |
| 45      | 129,818.39     | 3                 | 9,217.55             |
| 45      | 120,591.19     | 1                 | 195,123.94           |
| 60      | 0.00           | 1                 | 17,978.68            |
```

### Observation:

```
- High Risk customers are typically between ages 45–65.
- Many maintain high account balances and show product diversification.
- These accounts represent high-value customers at elevated churn risk.
```
--- 

</details>

## Recommended Actions (Across All Analyses)

```
1. Target churn reduction efforts in Germany, the primary high-risk market.
2. Launch female-focused engagement strategies to address higher churn among women.
3. Develop retention programs for older customers, especially those aged 51–60.
4. Introduce credit-risk-tailored retention programs, particularly for Poor and Excellent credit groups.
5. Maintain strong engagement with customers aged 18–30, the lowest-risk demographic.
6. Prioritize Medium & High Risk customers for retention outreach.
7. Build age-targeted programs, especially for customers 45+.
8. Offer product bundle optimization or financial planning for customers with high balances.
9. Deploy proactive engagement for customers showing diverse product usage, indicating high lifetime value.
10. Maintain monitoring and early-warning triggers for rising churn probability.
```
## 💼 Portfolio Highlights

- **Data Analysis:** Cleaned and merged messy Excel data, handled missing values, and engineered new features.  
- **Data Visualization:** Created insightful charts highlighting churn trends by geography, age, gender, and credit score.  
- **Predictive Modeling:** Built and evaluated a Random Forest model with 86.7% accuracy, identifying key churn drivers.  
- **Business Insights:** Segmented customers into risk tiers and recommended actionable retention strategies.  
- **Communication:** Presented findings in a structured, visual, and narrative format suitable for stakeholders.

---

This analysis demonstrates a complete end-to-end approach to customer churn management, from data cleaning and visualization to predictive modeling and actionable business insights.

Data Source: [Maven Analytics](https://mavenanalytics.io/data-playground/bank-customer-churn)
