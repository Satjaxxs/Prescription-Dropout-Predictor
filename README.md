# Rx Retention AI Agent: Enhancing Patient Adherence & Mitigating Revenue Loss

This repository hosts the code and resources for an AI-powered agent designed to improve patient adherence to medication, forecast potential revenue loss due to non-adherence, and provide actionable business strategies for pharmaceutical and healthcare companies.

## Table of Contents ;

1.  [Project Overview](#1-project-overview)
2.  [Business Problem & Solution](#2-business-problem--solution)
3.  [Key Features](#3-key-features)
**4.  [What the AI Model Did](#4-what-the-ai-model-did)**
5.  [Core Concepts](#5-core-concepts)
    * [What is Adherence?](#what-is-adherence)
    * [High Risk of Adherence](#high-risk-of-adherence)
    * [Refill Gap Rates](#refill-gap-rates)
    * [Adherence Linked to Business](#adherence-linked-to-business)
    * [Financial Impact Analysis (Budget, Profit/Loss)](#financial-impact-analysis-budget-profitloss)
6.  [Formulas and Mathematical Relations](#6-formulas-and-mathematical-relations)
    * [Financial Impact Formulas](#financial-impact-formulas)
    * [Statistical Formulas (for Model Evaluation)](#statistical-formulas-for-model-evaluation)
    * [Mathematical Relations for Machine Learning Models](#mathematical-relations-for-machine-learning-models)
7.  [Project Workflow](#7-project-workflow)
    * [Section 1: Data Preparation & SQL Power](#section-1-data-preparation--sql-power)
    * [Section 2: AI Prediction Engine](#section-2-ai-prediction-engine)
    * [Section 3: Unique Angles & Business Impact](#section-3-unique-angles--business-impact)
8.  [Business Strategy](#8-business-strategy)

---

## 1. Project Overview

The Rx Retention AI Agent is a Streamlit-based interactive application that demonstrates a comprehensive data science workflow. It tackles the critical challenge of patient non-adherence in medication by building predictive models, segmenting patient behavior, and quantifying the financial implications for pharmaceutical and healthcare companies. The agent provides actionable insights and strategic recommendations to mitigate revenue loss and improve patient outcomes.

## 2. Business Problem & Solution

**The Problem:**
Patient non-adherence to prescribed medication regimens is a pervasive issue in healthcare. This directly leads to:
* Suboptimal health outcomes for patients.
* **Significant revenue loss** for pharmaceutical companies and pharmacies due to missed refills and discontinuation of therapy. This is quantified as "Total Estimated Annual Revenue Loss Due to Non-Adherence".

**The Solution (Our AI Agent):**
This project provides an AI-driven solution designed to:
* **Identify patients at high risk of non-adherence (dropout)**.
* **Understand *when* these patients are likely to drop out** (Time-Aware Prediction).
* **Categorize patients into distinct behavioral segments** to understand their unique needs and motivations for non-adherence (Behavioral Segmentation).
* **Quantify the financial impact** (potential revenue loss) associated with non-adherence, providing a clear business case for intervention.
* **Offer actionable, data-driven strategies** to improve adherence and optimize financial outcomes.

## 3. Key Features

* **Data Preparation & Enhancement:** Automated loading, cleaning, and transformation of patient data, including synthetic balancing of the target variable for robust model training.
* **Simulated Feature Engineering:** Creation of granular behavioral features (e.g., `Refill_Gap_Days`, `No_of_Refills`, `Total_Months_on_Drug`, `Number_of_chronic_conditions`, `Average_Fills_per_Month`) to provide deeper insights.
* **SQL Analytics Integration:** Initial exploratory data analysis using SQL queries on the prepared dataset.
* **Binary Classification Models:** Training and evaluation of Logistic Regression and XGBoost classifiers to predict patient adherence.
* **Overfitting Mitigation:** Implementation of K-Fold Cross-Validation and Hyperparameter Tuning (using GridSearchCV) for robust model training and performance estimation.
* **Time-Aware Prediction:** Identification of critical stages in a patient's treatment journey where dropout risk is highest.
* **Behavioral Segmentation:** Clustering of patients into distinct groups based on their medication adherence patterns and health characteristics.
* **Financial Impact Estimation:** Calculation of potential revenue loss due to non-adherence, broken down by age group and behavioral segments.
* **Interactive Budget Analysis:** Allows users to input a budget for adherence initiatives and see the projected net gain/loss based on a mitigation assumption.
* **Strategic Recommendations:** Provides concrete, data-driven suggestions for business improvement, tailored to identified problems and segments.

---

## 4. What the AI Model Did

The AI model within the Rx Retention AI Agent performed several crucial functions to transform raw patient data into actionable insights and predictions:

* **Learned from Patient Data:**
    * It took in all the various characteristics (features) of patients and their prescription history (the 72 columns like `Refill_Gap_Days`, `No_of_Refills`, `Total_Months_on_Drug`, `Number_of_chronic_conditions`, etc.).
    * It then learned patterns from this data, specifically how these characteristics relate to whether a patient ended up being "Adherent" or "Not Adherent".

* **Predicted Patient Adherence:**
    * Using the patterns it learned, the model's primary task was to predict whether a new or existing patient is likely to become "Not Adherent" (i.e., stop taking their medication as prescribed).
    * It outputs not just a "yes/no" prediction, but also a `Dropout_Risk_Probability`, which indicates how likely a patient is to drop out (a score between 0 and 1).

* **Identified High-Risk Patients:**
    * By predicting the `Dropout_Risk_Probability`, the AI model allowed the system to flag specific patients who are at a high risk of non-adherence, enabling proactive intervention.

* **Performed Time-Aware Prediction:**
    * It analyzed the dropout risk in relation to the `Total_Months_on_Drug`, helping to identify specific critical stages in a patient's treatment journey where they are most likely to stop adherence.

* **Segmented Patient Behaviors:**
    * Beyond just prediction, the model also used K-Means Clustering to group patients into distinct behavioral segments based on their adherence patterns and health characteristics. This helps understand different types of non-adherence.

* **Quantified Financial Impact:**
    * The model's predictions (specifically the `Dropout_Risk_Probability` for 'Not Adherent' patients) were used as a crucial input to calculate the potential revenue loss due to non-adherence. This directly links patient behavior to business finances.

---

## 5. Core Concepts

### What is Adherence?

In the context of medication, **adherence** refers to the extent to which a patient takes their medication as prescribed by their healthcare provider. This includes taking the correct dose, at the correct time, and for the prescribed duration. Non-adherence (or "Not Adherent" in our project's target variable) means the patient is deviating from this prescribed regimen.

### High Risk of Adherence

A **high risk of adherence** (or more accurately, **high risk of non-adherence**) refers to patients who are predicted by the machine learning model to be "Not Adherent" (i.e., they are likely to stop taking their medication as prescribed) and have a high associated "Dropout_Risk_Probability". The project uses a user-defined probability threshold (e.g., 0.5) to identify these high-risk individuals.

### Refill Gap Rates

"Refill Gap Rates" refer to the **number of days between a patient's successive prescription refills**.
* **Higher Refill Gap Days** generally indicate **less adherence**, as patients are taking longer to refill their prescriptions than expected.
* **Lower Refill Gap Days** suggest **better adherence**, implying consistent medication use.

### Adherence Linked to Business

Adherence is fundamentally linked to business outcomes primarily through **revenue and patient lifetime value**.
* **Direct Revenue:** Each medication refill generates revenue. Non-adherence directly leads to missed refills, which are lost sales.
* **Patient Lifetime Value:** Adherent patients typically have better health outcomes, stay on therapy longer, and contribute sustained revenue over time.
* **Customer Loyalty:** Effective adherence support programs enhance patient experience, fostering loyalty and positive brand reputation.

The AI Agent quantifies the financial impact of non-adherence (estimated revenue loss) to provide a clear business case for investing in adherence initiatives.

### Financial Impact Analysis (Budget, Profit/Loss)

This analysis quantifies the potential financial return of investing in adherence programs:
1.  **Estimated Revenue Loss:** The system calculates the total revenue expected to be lost if patients continue to be non-adherent, based on their dropout probabilities and assumed drug economics.
2.  **Budget Input:** The user provides a "Company's Budget for Adherence Initiatives" – the investment made to improve adherence.
3.  **Mitigation Assumption:** A business assumption (e.g., 30% mitigation) is applied: "If we spend money, we expect to successfully recover this percentage of lost revenue".
4.  **Net Impact (Profit/Loss):** The calculation then determines if the expected recovered revenue (mitigated loss) outweighs the investment (budget).
    * **Net Gain:** Recovered Revenue > Budget.
    * **Net Loss:** Recovered Revenue < Budget.
    * **Break-even:** Recovered Revenue = Budget.

This provides a direct financial justification for adherence initiatives, allowing businesses to make data-driven investment decisions.

## 6. Formulas and Mathematical Relations

#### Financial Impact Formulas

The financial impact analysis quantifies the potential revenue loss and the projected gain/loss from adherence initiatives.

* **Average Drug Price Per Month ($P$)**: Input by the user (e.g., ₹500).
* **Average Missed Months Per Dropout ($M$)**: Input by the user (e.g., 6 months).
* **Dropout Risk Probability ($P_{dropout}$)**: Predicted by the AI model for each patient (a value between 0 and 1).
* **Potential Revenue Loss per Patient ($L_i$)**:
    For each patient $i$ predicted as 'Not Adherent', the potential revenue loss is calculated as:
    $L_i = P_{dropout,i} \times P \times M$

* **Total Estimated Annual Revenue Loss ($L_{total}$)**:
    This is the sum of `Potential_Revenue_Loss` for all patients predicted as 'Not Adherent' (where `Predicted_Adherent` = 1):
    $L_{total} = \sum_{i \in \text{Predicted Not Adherent}} L_i$

* **Company Budget for Adherence Initiatives ($B$)**: Input by the user (e.g., ₹100,000).
* **Potential Loss Mitigation Percentage ($M_p$)**: A business assumption, typically 30% (0.30).
* **Mitigated Loss Amount ($L_{mitigated}$)**:
    $L_{mitigated} = L_{total} \times M_p$

* **Projected Net Impact (Gain/Loss) ($Net_{impact}$)**:
    $Net_{impact} = L_{mitigated} - B$
    * If $Net_{impact} > 0$, it's a projected net gain.
    * If $Net_{impact} < 0$, it's a projected net loss.
    * If $Net_{impact} = 0$, it's a break-even.

#### Statistical Formulas (for Model Evaluation)

These metrics are used to assess the performance of the machine learning classification models.

First, let's define the components of a **Confusion Matrix** for a binary classification problem (Adherent (0) vs. Not Adherent (1)):
* **True Positive (TP)**: Correctly predicted positive (Actual 1, Predicted 1).
* **True Negative (TN)**: Correctly predicted negative (Actual 0, Predicted 0).
* **False Positive (FP)**: Incorrectly predicted positive (Actual 0, Predicted 1). This is a Type I error.
* **False Negative (FN)**: Incorrectly predicted negative (Actual 1, Predicted 0). This is a Type II error.

* **Accuracy**: The proportion of correctly predicted instances over the total number of instances.
    $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$

* **Precision (for class 1 - Not Adherent)**: The proportion of correctly predicted positive observations out of all observations predicted as positive.
    $Precision = \frac{TP}{TP + FP}$

* **Recall (Sensitivity, for class 1 - Not Adherent)**: The proportion of correctly predicted positive observations out of all actual positive observations.
    $Recall = \frac{TP}{TP + FN}$

* **F1-Score (for class 1 - Not Adherent)**: The harmonic mean of Precision and Recall. It's a good metric for imbalanced datasets.
    $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$

* **Standard Deviation ($\sigma$) for Cross-Validation Scores**: Measures the dispersion or spread of the cross-validation scores around their mean.
    Given a set of $n$ scores $x_1, x_2, \dots, x_n$ and their mean $\bar{x}$:
    $\sigma = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n}}$
    This value indicates the variability or stability of the model's performance across different data folds.

#### Mathematical Relations for Machine Learning Models

Your project uses Logistic Regression, XGBoost, and K-Means.

* **a) Logistic Regression (for Binary Classification)**
    Logistic Regression is a linear model for binary classification. It models the probability that a given input belongs to a certain class.
    * **Linear Combination**: For a given input $\mathbf{x} = [x_1, x_2, \dots, x_m]$, it calculates a linear combination of features:
        $z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_m x_m = \mathbf{w}^T \mathbf{x} + b$
        where $\mathbf{w}$ are the weights (coefficients), $\mathbf{x}$ are the features, and $b$ (or $w_0$) is the bias (intercept).
    * **Sigmoid Function**: This linear output $z$ is then passed through a sigmoid (or logistic) activation function to map it to a probability between 0 and 1:
        $\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$
        $\hat{p}$ is the predicted probability that the instance belongs to class 1 (Not Adherent).
    * **Prediction**: The model predicts class 1 if $\hat{p} \geq 0.5$, and class 0 otherwise.
    * **Cost Function (Binary Cross-Entropy Loss)**: Logistic Regression aims to minimize a cost function that penalizes incorrect probability predictions. For a single training instance:
        $J(\mathbf{w}) = - [y \log(\hat{p}) + (1 - y) \log(1 - \hat{p})]$
        where $y$ is the actual label (0 or 1). The total cost is the average over all training instances.
    * **Regularization**: The `C` parameter in `LogisticRegression` (used in your code) is the inverse of regularization strength. Smaller values of `C` imply stronger regularization (L2 by default, or L1 if `penalty='l1'`), which helps prevent overfitting by penalizing large weights.

* **b) XGBoost (Extreme Gradient Boosting Classifier)**
    XGBoost is an ensemble learning method that builds a strong predictive model by combining a sequence of weak prediction models (typically decision trees). It is an optimized distributed gradient boosting library.
    * **Gradient Boosting**: It builds trees sequentially, with each new tree trying to correct the errors of the previous ones. It minimizes a *loss function* by adding new models that predict the residuals (the differences between actual and predicted values).
    * **Objective Function**: XGBoost optimizes an objective function that combines a training loss component (measuring predictive accuracy) and a regularization component (to control complexity and prevent overfitting).
        $Obj = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)$
        where $l$ is the differentiable convex loss function, $\hat{y}_i^{(t)}$ is the prediction at iteration $t$, and $\Omega(f_k)$ is the regularization term for the $k$-th tree $f_k$.
    * **Regularization**: The regularization term $\Omega(f_k)$ includes L1 ($\alpha$) and L2 ($\lambda$) regularization parts on the tree weights and a term for the number of leaves, which penalizes complex trees. This is crucial for controlling overfitting.
    * **Boosting Process**: The algorithm iteratively adds trees, where each tree is fit to the negative gradient of the loss function (also known as pseudo-residuals) from the previous ensemble.
    * **Hyperparameters**: Parameters like `n_estimators` (number of trees), `learning_rate` (step size shrinkage to prevent overfitting), and `max_depth` (depth of individual trees) are tuned to optimize performance and prevent overfitting.

* **c) K-Means Clustering**
    K-Means is an unsupervised learning algorithm used for clustering data points into $K$ distinct clusters.
    * **Objective**: The goal of K-Means is to partition $n$ data points into $K$ distinct clusters, where each data point belongs to the cluster with the nearest mean (centroid).
    * **Minimization**: It aims to minimize the within-cluster sum of squares (WCSS), also known as inertia.
        $WCSS = \sum_{j=1}^{K} \sum_{i=1}^{n_j} ||x_i - c_j||^2$
        where $K$ is the number of clusters, $n_j$ is the number of data points in cluster $j$, $x_i$ is a data point, and $c_j$ is the centroid of cluster $j$.
    * **Algorithm (Iterative Process)**:
        1.  **Initialization**: Randomly select $K$ data points as initial centroids.
        2.  **Assignment Step**: Assign each data point to the nearest centroid.
        3.  **Update Step**: Recalculate the centroids as the mean of all data points assigned to that cluster.
        4.  **Repeat**: Steps 2 and 3 are repeated until the centroids no longer change significantly, or a maximum number of iterations is reached.
    * **Standardization**: K-Means is sensitive to the scale of features, so `StandardScaler` is applied to the clustering features before applying K-Means. This ensures that features with larger numerical ranges do not disproportionately influence the distance calculations.

## 6. Project Workflow

The Streamlit application guides the user through three main sections:

### Section 1: Data Preparation & SQL Power

* **Objective**: Ingest raw data, clean and transform it, create new features, and make it ready for analysis and machine learning.
* **Steps**:
    1.  Load `Persistent_vs_NonPersistent.csv`.
    2.  Rename columns and transform `Persistency_Flag` to `Adherent` (1=Not Adherent, 0=Adherent).
    3.  **Crucially, artificially balance the `Adherent` target variable** by flipping a percentage of 'Adherent' patients to 'Not Adherent' to improve model training for the minority class.
    4.  Simulate and add granular features like `Refill_Gap_Days`, `No_of_Refills`, `Total_Months_on_Drug`, `Number_of_chronic_conditions`, `Average_Fills_per_Month`, etc., based on `Adherent` status.
    5.  Save the enhanced dataset to `patient_data_enhanced.csv`.
    6.  Load the enhanced data into an SQLite database (`rx_retention_enhanced.sqlite`).
    7.  Execute SQL queries to derive initial insights (e.g., average refill gaps by adherence, non-adherence trends by months on drug).

### Section 2: AI Prediction Engine

* **Objective**: Build, train, and evaluate machine learning models to predict patient adherence.
* **Steps**:
    1.  Load the `patient_data_enhanced.csv` dataset.
    2.  Define features (X) and target (y - `Adherent`).
    3.  Identify numerical and categorical features for preprocessing.
    4.  Set up a `ColumnTransformer` for data preprocessing (StandardScaling for numerical, OneHotEncoding for categorical).
    5.  Split data into training and testing sets (75/25 split, stratified).
    6.  **Train and tune two classification models (Logistic Regression and XGBoost Classifier)**.
        * **K-Fold Cross-Validation (e.g., 5-fold) is integrated with `GridSearchCV`** for hyperparameter tuning. This ensures robust performance evaluation and helps mitigate overfitting by training/testing on different data subsets.
    7.  Evaluate models using accuracy, F1-score, confusion matrix, and classification reports on the test set. Average cross-validation scores are also presented for a more reliable performance estimate.
    8.  The best-performing model (based on F1-score from cross-validation) is saved as `rx_retention_classifier_model.pkl` for future use.

### Section 3: Unique Angles & Business Impact

* **Objective**: Apply the trained model to generate advanced insights and estimate financial impact.
* **Steps**:
    1.  Load the enhanced dataset and the best-trained classifier model.
    2.  Generate predictions (`Predicted_Adherent` and `Dropout_Risk_Probability`) for all patients.
    3.  **Time-Aware Prediction**:
        * Identify "high-risk" patients based on predicted non-adherence and a user-defined probability threshold.
        * Analyze how average dropout risk changes with 'Total_Months_on_Drug', identifying critical intervention points (e.g., early months).
    4.  **Behavioral Segmentation (K-Means Clustering)**:
        * Cluster patients into distinct behavioral segments (number of clusters is user-adjustable) using key features like refill behavior, chronic conditions, and dropout risk.
        * Characterize each segment with its average feature values, allowing for clear interpretation (e.g., "High-Risk, Inconsistent Refillers", "Low-Risk, Highly Adherent").
        * Visualize cluster distribution using pie charts and scatter plots.
    5.  **Financial Impact Estimation**:
        * Calculate the `Potential_Revenue_Loss` for each predicted non-adherent patient using user-defined `Average Drug Price Per Month` and `Average Missed Months Per Dropout`.
        * Sum these to get the `Total Estimated Annual Revenue Loss Due to Non-Adherence`.
        * Break down this loss by `Age_Group` and `Behavioral_Cluster`.
        * **Interactive Budget Analysis**:
            * User inputs a `Company's Budget for Adherence Initiatives`.
            * A `potential_loss_percentage` (e.g., 30%) is assumed to be mitigated by this budget.
            * Calculate `Projected Net Gain/Loss` (Mitigated Loss - Budget).
    6.  **Strategic Recommendations**: Provide actionable business improvement suggestions, directly linking them to identified problems and the financial analysis.

**Business Strategy:**
The synthesis of these data points enables a powerful, data-driven business strategy:
1.  **Identify At-Risk Patients:** Leveraging the rx_retention_classifier_model, patients with a high probability of non-persistence can be flagged.
2.  **Targeted Intervention:** Resources for adherence initiatives ($C_{int}$) can then be strategically focused on these identified high-risk patients, rather than being disbursed indiscriminately. Examples of interventions include personalized medication reminders, patient education programs, follow-up calls from pharmacists or care coordinators, and simplified refill processes.
3.  **Convert Non-Persistent to Persistent:** The ultimate goal is that a successful intervention changes a patient's behavior from being at-risk of non-persistence to becoming persistent. In such cases, the incremental gain in Profit_{Persistent} from continued refills far outweighs the cost of the intervention ($C_{int}$).
4.  **Optimize Return on Investment (ROI):** The success of this strategy is measured by its ability to maximize the overall net gain. The objective is to ensure that the sum of (Profit_{Persistent} - C_{int}) for successfully intervened patients, combined with the continued profits from naturally persistent patients, significantly outweighs the LostProfit_{NonPersistent} from those who still drop out and the costs of failed interventions.
    $$Optimize \ ROI = \text{Maximize} \left( \sum_{\text{Successfully Converted Patients}} (Profit_{Persistent} - C_{int}) + \sum_{\text{Naturally Persistent Patients}} Profit_{Persistent} \right) - \sum_{\text{Lost Profit from Non-Persistent}} - \sum_{\text{Costs of Failed Interventions}}$$
By implementing this AI-powered approach, businesses can move from reactive problem-solving to proactive patient management, leading to improved financial performance and, critically, better health outcomes for patients.
