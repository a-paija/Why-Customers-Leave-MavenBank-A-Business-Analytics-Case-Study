# Customer Churn Analysis & Retention Strategy

## 📊 Business Case Scenario
Mavenland Bank is facing an unexpected rise in customer churn that is threatening quarterly revenue targets. The executive team from Product, Marketing, and Customer Success departments have convened an emergency meeting and need your analytical expertise to understand and address this critical issue.

**Business Objective:** Reduce customer churn by 25% within two quarters to stabilize quarterly revenue targets.

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

## 💡 Actionable Recommendations

| **Immediate Priority Actions** | **Strategic Initiatives** |
|-------------------------------|---------------------------|
| **German Customer Retention Program** <br> - Dedicated German-speaking relationship managers <br> - Market-specific product bundles <br> - Cultural sensitivity training for staff | **High-Value Customer Protection** <br> - Proactive outreach to high-balance customers <br> - Exclusive loyalty benefits and premium services <br> - Personalized financial reviews |
| **Age-Specific Retention Strategy** <br> - Target customers aged 50-60 with personalized financial planning <br> - Develop retirement-focused product offerings <br> - High-touch service for senior customers | **Predictive Retention System** <br> - Implement churn risk scoring in CRM <br> - Automated alerts for high-risk customers <br> - Dynamic campaign triggering based on risk tiers |
| **Product Engagement Initiative** <br> - Cross-selling campaigns for single-product customers <br> - Bundle discounts for multiple products <br> - Frontline staff incentives for product adoption | |
  
## 🔍 Exploratory Analysis & Insights

<details>
<summary><strong>Objective 1: Overall Churn Rate</strong></summary>
</details>

