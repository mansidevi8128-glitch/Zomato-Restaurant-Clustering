# Zomato Restaurant Clustering & Sentiment Analysis (Hyderabad)

Unsupervised machine learning project, done as part of my **Labmentix Data Science internship**. I grouped Zomato restaurants in Gachibowli, Hyderabad into business segments and built a model that classifies customer reviews as positive or negative.

---

## Problem Statement

Zomato has thousands of restaurants and a huge number of free-text reviews, but it has no simple, data-driven way to group similar restaurants or to tell automatically whether a customer is happy. This project aims to:

1. **Cluster restaurants** into meaningful segments based on cost, cuisine mix, popularity and ratings.
2. **Analyse customer sentiment** in reviews and find what drives good and bad experiences.
3. Do a **cost vs. benefit analysis**: is a higher price justified by a higher rating?
4. Use reviewer metadata to **identify critics and influential reviewers**.
5. Give recommendations that help **customers** find the best restaurant and help **Zomato** improve where it is lagging.

---

## Dataset

| File | Rows | Description |
|---|---|---|
| `Zomato_Restaurant_names_and_Metadata.csv` | 105 restaurants | Name, Links, Cost (for two), Collections, Cuisines, Timings |
| `Zomato_Restaurant_reviews.csv` | 10,000 reviews | Restaurant, Reviewer, Review, Rating, Metadata (reviews/followers), Time, Pictures |

The reviews cover May 2016 to May 2019. 100 restaurants have exactly 100 reviews each, and 5 restaurants have no reviews.

---

## Project Workflow

```
EDA  ->  Data Cleaning  ->  Feature Engineering  ->  Preprocessing  ->  Modelling  ->  Explainability
```

1. **Data wrangling:** converted Cost and Rating to numbers, removed 36 blank duplicate rows, parsed reviewer metadata into reviews/followers, split timestamps, and aggregated reviews per restaurant (9,954 clean reviews).
2. **EDA:** 15+ charts following the UBM rule (Univariate, Bivariate, Multivariate).
3. **Hypothesis testing:** Spearman correlation, Mann-Whitney U and Chi-square tests.
4. **Feature engineering:** IQR capping of outliers, multi-label encoding of cuisines, 5 cuisine-family share features, removal of correlated features, log transform, StandardScaler, and PCA (11 features → 7 components, 90% variance).
5. **NLP pipeline:** contraction expansion, lower-casing, punctuation/URL/digit removal, stop-word removal (negations kept), slang normalisation, tokenisation, lemmatisation, and TF-IDF (unigrams + bigrams).
6. **Models:**
   - K-Means clustering (tuned with elbow, silhouette grid search and a stability check)
   - Agglomerative (hierarchical) clustering with a dendrogram
   - Sentiment classifier: TF-IDF + Logistic Regression, tuned with 5-fold GridSearchCV
7. **Explainability:** surrogate Random Forest with permutation importance for the clusters, and word weights for the sentiment model.
8. **Deployment readiness:** models saved with joblib, reloaded, and tested on unseen data.

---

## Key Insights

- The median **cost for two is Rs 700**. **North Indian (61) and Chinese (43)** are by far the most common cuisines.
- Ratings are polarised: **63% positive vs 24.6% negative**, and unhappy customers usually give 1★.
- **Cost and rating are positively related** (Spearman ρ = 0.38, p = 0.0001). Negative reviews fall from **32% for budget** restaurants to **16% for premium** ones.
- **Mediterranean (4.55) and Asian (3.92)** cuisines are rated highest. **Fast Food, Andhra and Biryani** are rated lowest.
- Review activity peaks on **weekends between 9 and 11 PM**.
- **Critics (1,000+ followers)** write about 3.6× longer reviews and are less negative than casual reviewers.
- Negative reviews mostly mention **orders, delivery time, cold or stale food and taste**. Positive ones mention **ambience, service and staff**.

---

## Restaurant Segments (K-Means, k = 4)

| Segment | Restaurants | Avg Cost | Avg Rating | Examples |
|---|---|---|---|---|
| Premium Continental/Multi-cuisine – high-rated | 43 | Rs 1,294 | 3.84 | AB's, Flechazo, Over The Moon, Feast – Sheraton |
| Mid-priced Indian – low-rated | 38 | Rs 611 | 3.39 | Paradise, Pakwaan Grand, Behrouz Biryani |
| Budget Cafe & Desserts | 16 | Rs 497 | 3.62 | Cream Stone, Karachi Bakery, Labonel |
| Budget Quick-bites – low-rated | 8 | Rs 456 | 3.32 | KFC, Domino's, GD's |

Silhouette score 0.284, Davies-Bouldin 1.235, and identical segments across 10 random seeds (ARI = 1.0).

---

## Model Results

| Model | Metric | Score |
|---|---|---|
| K-Means (final clustering model) | Silhouette | **0.284** |
| Agglomerative (Ward) | Silhouette | 0.249 |
| Sentiment – Logistic Regression (tuned) | Accuracy | **94.3%** |
| | F1-score | **0.960** |
| | ROC-AUC | **0.984** |

---

## Business Recommendations

- **Premium high-rated segment:** promote it in *Top-Rated* and *Great Buffets* collections and use it for premium offers.
- **Mid-priced Indian, low-rated (largest problem segment):** run quality and delivery improvement programmes and give these restaurants review-response tools.
- **Budget Cafe & Desserts:** run weekend and evening snack campaigns.
- **Budget Quick-bites:** focus on delivery speed and packaging.
- Use the sentiment model to **flag negative reviews in real time**, so restaurants can respond before the customer churns.

---

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · SciPy · Scikit-learn · Joblib

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Zomato-Restaurant-Clustering.git
   cd Zomato-Restaurant-Clustering
   ```
2. Install the libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn joblib
   ```
3. Keep both CSV files in the same folder as the notebook.
4. Open `Zomato_Restaurant_Clustering_Submission.ipynb` in Jupyter or Google Colab and click **Run All**. The notebook runs end to end without errors.

---

## Repository Structure

```
├── Zomato_Restaurant_Clustering_Submission.ipynb   # Complete analysis notebook
├── Zomato_Restaurant_names_and_Metadata.csv        # Restaurant data
├── Zomato_Restaurant_reviews.csv                   # Review data
└── README.md
```

---

## Author

**[Your Name]**
Data Science Intern, Labmentix
[LinkedIn](https://www.linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)
