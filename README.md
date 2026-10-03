# SmartCart Clustering System 🛒

Intelligent customer segmentation for e-commerce using unsupervised learning.

### 🔗 Live Repo: https://github.com/AratiLikitha/smartcart-clustering-system

## 📌 Problem Statement
SmartCart is a growing e-commerce platform serving customers across multiple countries. The company collected extensive data of 2240 customers and 29 attributes.

Currently, SmartCart uses generic marketing strategies for all customers, without understanding different customer behaviour patterns. This results in inefficient marketing and poor retention.

**Objective:** Build an intelligent customer segmentation system using unsupervised ML to discover meaningful clusters based on income, spending, purchase behaviour and website activity.

## 📂 Dataset Description
Each row represents one customer.

**1. Customer Demographics:** ID, Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome, Dt_Customer

**2. Purchase Amount (Mnt):** MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds

**3. Purchase Behaviour (Frequency):** NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth

**4. Customer Response:** Recency, Complain, Response

Source: `smartcart_customers.csv` (2240 x 29)

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
- **ML Algorithms:** K-Means Clustering
- **Concepts:** PCA, Elbow Method, Silhouette Score, StandardScaler
- **Tool:** Jupyter Notebook

## 🔄 My Approach

**1. Data Preprocessing**
- Checked shape (2240, 29), handled 24 missing Income values with median
- Converted Dt_Customer to datetime, calculated customer tenure

**2. Feature Engineering**
- `Age = 2025 - Year_Birth`
- `Total_Spending = Sum of all Mnt columns`
- `Total_Children = Kidhome + Teenhome`
- `Family_Size, Is_Parent, Total_Purchases`

**3. EDA & Visualization**
- Spending pattern analysis by Education & Marital Status
- Correlation heatmap between Income and Spending
- Distribution of Web Visits vs Purchases

**4. Encoding & Scaling**
- One-Hot Encoding for Education, Marital_Status
- StandardScaler for numerical features

**5. Dimensionality Reduction**
- Applied PCA to reduce to 2 components for cluster visualization
- Explained variance ratio analysis

**6. Finding Optimal K**
- **Elbow Method:** WCSS vs K -> Elbow at K=4
- **Silhouette Analysis:** Highest score ~0.42 at K=4

**7. Final Clustering**
- Applied K-Means with K=4 and visualized clusters with scatter plot

## 📊 Clusters Identified

| Cluster | Segment Name | Characteristics |
| :--- | :--- | :--- |
| 0 | **High-Value Loyalists** | High Income (>80k), High Spending on Wines & Meat, Low Recency |
| 1 | **Budget-Conscious Families** | Have kids, Low Income, High NumDealsPurchases |
| 2 | **At-Risk Customers** | High Recency (50+ days), Low spending, Need re-engagement |
| 3 | **Web-Savvy Shoppers** | Young age, High NumWebPurchases & NumWebVisitsMonth |

## ▶️ How to Run
```bash
git clone https://github.com/AratiLikitha/smartcart-clustering-system.git
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook smartcart.ipynb
