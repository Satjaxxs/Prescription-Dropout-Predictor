# --- Section 4: Conversational Dashboard (Gradio App) ---

import gradio as gr
import pandas as pd
import numpy as np
import os
import joblib # For loading the trained model
import matplotlib.pyplot as plt
import seaborn as sns
import shap # For SHAP values (model explainability)
from sklearn.preprocessing import StandardScaler # Required for the workaround

print("--- Starting Gradio App for RxRetention AI ---")

# --- Global Configurations & File Paths ---
DATA_PATH = os.path.join('data', 'patient_data_enhanced.csv')
MODEL_PATH = os.path.join('models', 'rx_retention_classifier_model.pkl')

# --- Helper Functions for Data/Model Loading (Cached for Performance) ---
# @gr.cached allows caching results of functions to improve performance.
@gr.memo()
def load_data():
    """Loads the enhanced patient data from CSV."""
    try:
        df = pd.read_csv(DATA_PATH)
        if 'Adherent' in df.columns:
            df['Adherent'] = df['Adherent'].astype(int)
        print("DEBUG: Data loaded successfully from CSV.")
        return df
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_PATH}. Please ensure Section 1 was run correctly.")
        return None
    except Exception as e:
        print(f"ERROR: An error occurred during data loading: {e}")
        return None

# MODIFIED: Force re-fitting of StandardScaler if not fitted after loading
# This is the aggressive workaround for the persistent 'StandardScaler is not fitted yet' error.
@gr.memo() # Use Gradio's caching mechanism
def load_model_pipeline():
    """
    Loads the trained ML pipeline and forcefully replaces/fits its StandardScaler component.
    This is an aggressive workaround for persistent 'StandardScaler is not fitted yet' errors.
    """
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print("DEBUG: Model pipeline loaded successfully by joblib.")

        preprocessor = model_pipeline.named_steps['preprocessor']
        print(f"DEBUG: Preprocessor steps: {preprocessor.transformers}")

        # --- AGGRESSIVE WORKAROUND FOR 'StandardScaler is not fitted yet' ERROR ---
        # This will forcefully re-fit the numerical scaler component every time the model is loaded.
        
        numerical_transformer_name = 'num'
        numerical_transformer_index = -1
        
        for i, (name, transformer, cols) in enumerate(preprocessor.transformers):
            if name == numerical_transformer_name:
                numerical_transformer_index = i
                original_name = name 
                original_cols = cols
                break

        if numerical_transformer_index != -1:
            print("DEBUG: Found numerical transformer ('num'). Attempting forceful re-fit.")
            
            temp_df_for_fit = load_data() 
            if temp_df_for_fit is None:
                print("ERROR: Failed to load data for forceful re-fitting of StandardScaler. Stopping.")
                return None
            
            numerical_features_for_fitting = [
                'Refill_Gap_Days', 'No_of_Refills', 'Total_Months_on_Drug',
                'Dexa_Freq_During_Rx', 'Count_Of_Risks', 'Number_of_chronic_conditions',
                'Number_of_medications', 'Average_Fills_per_Month', 'Average_Days_Supply',
                'Average_Refills_per_Prescription'
            ]
            numerical_features_for_fitting = [col for col in numerical_features_for_fitting if col in temp_df_for_fit.columns]

            if not numerical_features_for_fitting:
                print("ERROR: Could not find valid numerical features to re-fit StandardScaler. SHAP/Predictions might fail.")
                return None

            # Fill any potential NaNs in the data used for fitting
            for col in numerical_features_for_fitting:
                if temp_df_for_fit[col].isnull().any():
                    temp_df_for_fit[col] = temp_df_for_fit[col].fillna(temp_df_for_fit[col].median())
            
            try:
                # Create a brand new StandardScaler and fit it
                print("DEBUG: Creating and fitting a NEW StandardScaler instance.")
                new_scaler = StandardScaler()
                new_scaler.fit(temp_df_for_fit[numerical_features_for_fitting])
                print("DEBUG: New StandardScaler instance successfully created and fitted.")

                # Replace the old numerical transformer object with the newly fitted one
                transformers_list = list(preprocessor.transformers)
                transformers_list[numerical_transformer_index] = (original_name, new_scaler, original_cols)
                preprocessor.transformers = transformers_list
                model_pipeline.named_steps['preprocessor'] = preprocessor
                print("DEBUG: Pipeline preprocessor updated with NEW, re-fitted StandardScaler.")

            except Exception as fit_error:
                print(f"ERROR: Failed to re-fit StandardScaler: {fit_error}")
                return None
        else:
            print("DEBUG: Numerical transformer 'num' not found or is not an estimator with a fit method. Proceeding without re-fit.")

        # --- END ULTIMATE WORKAROUND ---

        return model_pipeline
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Please ensure Section 2 was run correctly and the model was saved.")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during model loading or re-fitting: {e}")
        return None

@gr.memo()
def get_transformed_feature_names(_pipeline_model, original_X_df_sample):
    """
    Extracts the feature names after preprocessing by the ColumnTransformer.
    This is critical for SHAP plots.
    """
    preprocessor = _pipeline_model.named_steps['preprocessor']

    numerical_transformer_name = 'num'
    numerical_feature_names = []
    for name, transformer, cols in preprocessor.transformers:
        if name == numerical_transformer_name:
            if hasattr(transformer, 'get_feature_names_out'):
                numerical_feature_names = transformer.get_feature_names_out(cols)
            elif cols:
                numerical_feature_names = [c for c in cols if c in original_X_df_sample.columns]
            break
    
    categorical_transformer_name = 'cat'
    ohe_feature_names = []
    for name, transformer, cols in preprocessor.transformers:
        if name == categorical_transformer_name:
            if hasattr(transformer, 'get_feature_names_out'):
                ohe_feature_names = transformer.get_feature_names_out(cols)
            elif cols:
                ohe_feature_names = [c for c in cols if c in original_X_df_sample.columns]
            break

    all_transformed_features = list(numerical_feature_names) + list(ohe_feature_names)
    
    return all_transformed_features


# --- Global Data and Model Instances (Loaded Once) ---
df = load_data()
model_pipeline = load_model_pipeline()
all_transformed_features = [] # Initialize; will be set after X_for_prediction_df is created

if df is None or model_pipeline is None:
    print("FATAL ERROR: Data or Model not loaded. Gradio app will not function.")
    # In a real app, you might want to raise an exception here or have a more graceful fallback
    # For now, just continue with empty data/model, which will cause further errors but allows code to run.
else:
    # Prepare DataFrame with Predictions (Executed once at startup)
    try:
        X_for_prediction_df = df.drop(['Patient_ID', 'Adherent'], axis=1)

        # Ensure prediction columns are not created if already exist (e.g. on hot reload)
        if 'Predicted_Adherent' not in df.columns:
            df['Predicted_Adherent'] = model_pipeline.predict(X_for_prediction_df)
        if 'Dropout_Risk_Probability' not in df.columns:
            df['Dropout_Risk_Probability'] = model_pipeline.predict_proba(X_for_prediction_df)[:, 1]

        # Get transformed feature names for SHAP (using a sample for caching)
        all_transformed_features = get_transformed_feature_names(model_pipeline, X_for_prediction_df.sample(min(100, len(X_for_prediction_df)), random_state=42))
        print("DEBUG: Predictions generated and feature names retrieved.")

    except Exception as e:
        print(f"FATAL ERROR: During initial prediction calculation or feature name retrieval: {e}")
        # This error likely means the model or preprocessor is still not working correctly.
        # Gradio app might not be fully functional.


# --- Gradio UI Functions ---

def get_overview_content():
    if df is None:
        return gr.Markdown("## Error: Data not loaded. Cannot display Overview.")
    
    total_patients = len(df)
    predicted_non_adherent = df['Predicted_Adherent'].sum()
    adherence_rate = (1 - df['Predicted_Adherent'].mean()) * 100

    metrics_markdown = f"""
    ## Dashboard Overview
    | Metric                       | Value                   |
    | :--------------------------- | :---------------------- |
    | Total Patients Analyzed      | **{total_patients}** |
    | Predicted At-Risk (Non-Adherent) | **{predicted_non_adherent}** |
    | Predicted Adherence Rate     | **{adherence_rate:.2f}%** |
    """

    # Predicted Non-Adherence Trends by Total Months on Drug
    non_adherence_by_month = df[df['Predicted_Adherent'] == 1].groupby('Total_Months_on_Drug').size().reset_index(name='Not_Adherent_Count')
    total_by_month = df.groupby('Total_Months_on_Drug').size().reset_index(name='Total_Patients')
    merged_data = pd.merge(non_adherence_by_month, total_by_month, on='Total_Months_on_Drug', how='right').fillna(0)
    merged_data['Non_Adherence_Percentage'] = (merged_data['Not_Adherent_Count'] / merged_data['Total_Patients']) * 100

    fig_time_trend, ax_time_trend = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Total_Months_on_Drug', y='Non_Adherence_Percentage', data=merged_data, marker='o', ax=ax_time_trend)
    ax_time_trend.set_title('Predicted Non-Adherence Percentage by Total Months on Drug')
    ax_time_trend.set_xlabel('Total Months on Drug')
    ax_time_trend.set_ylabel('Predicted Non-Adherence (%)')
    ax_time_trend.grid(True)
    
    plt.tight_layout()
    return metrics_markdown, fig_time_trend

def get_adherence_insights_content():
    if df is None:
        return [gr.Markdown("## Error: Data not loaded. Cannot display Adherence Insights.")] * 3
    
    figs = []

    # Non-Adherence by Age Group
    non_adherence_by_age = df.groupby('Age_Group')['Predicted_Adherent'].value_counts(normalize=True).unstack(fill_value=0)
    if 1 in non_adherence_by_age.columns:
        non_adherence_by_age['Non_Adherence_Percentage'] = non_adherence_by_age[1] * 100
        non_adherence_by_age = non_adherence_by_age.sort_values(by='Non_Adherence_Percentage', ascending=False)
        fig_age, ax_age = plt.subplots(figsize=(10, 6))
        sns.barplot(x=non_adherence_by_age.index, y=non_adherence_by_age['Non_Adherence_Percentage'], palette='viridis', ax=ax_age)
        ax_age.set_title('Predicted Non-Adherence Percentage by Age Group')
        ax_age.set_xlabel('Age Group')
        ax_age.set_ylabel('Non-Adherence (%)')
        ax_age.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        figs.append(fig_age)
    else:
        figs.append(gr.Markdown("No 'Not Adherent' predictions for age group analysis."))

    # Non-Adherence by Gender
    non_adherence_by_gender = df.groupby('Gender')['Predicted_Adherent'].value_counts(normalize=True).unstack(fill_value=0)
    if 1 in non_adherence_by_gender.columns:
        non_adherence_by_gender['Non_Adherence_Percentage'] = non_adherence_by_gender[1] * 100
        fig_gender, ax_gender = plt.subplots(figsize=(6, 4))
        sns.barplot(x=non_adherence_by_gender.index, y=non_adherence_by_gender['Non_Adherence_Percentage'], palette='plasma', ax=ax_gender)
        ax_gender.set_title('Predicted Non-Adherence Percentage by Gender')
        ax_gender.set_xlabel('Gender')
        ax_gender.set_ylabel('Non-Adherence (%)')
        plt.tight_layout()
        figs.append(fig_gender)
    else:
        figs.append(gr.Markdown("No 'Not Adherent' predictions for gender analysis."))

    # Non-Adherence by Count of Risks
    non_adherence_by_risk_count = df.groupby('Count_Of_Risks')['Predicted_Adherent'].value_counts(normalize=True).unstack(fill_value=0)
    if 1 in non_adherence_by_risk_count.columns:
        non_adherence_by_risk_count['Non_Adherence_Percentage'] = non_adherence_by_risk_count[1] * 100
        non_adherence_by_risk_count = non_adherence_by_risk_count.sort_index()
        fig_risk, ax_risk = plt.subplots(figsize=(10, 6))
        sns.barplot(x=non_adherence_by_risk_count.index, y=non_adherence_by_risk_count['Non_Adherence_Percentage'], palette='cividis', ax=ax_risk)
        ax_risk.set_title('Predicted Non-Adherence Percentage by Count of Risks')
        ax_risk.set_xlabel('Count of Risks')
        ax_risk.set_ylabel('Non-Adherence (%)')
        plt.tight_layout()
        figs.append(fig_risk)
    else:
        figs.append(gr.Markdown("No 'Not Adherent' predictions for risk count analysis."))
    
    return figs[0], figs[1], figs[2] # Return as separate outputs for Gradio

def get_shap_plot_content():
    if df is None or model_pipeline is None or not all_transformed_features:
        return gr.Markdown("## Error: Data or Model not loaded, or features not ready for SHAP.")
    
    X_raw = df.drop(['Patient_ID', 'Adherent'], axis=1) # Use the full X_for_prediction_df
    X_sample_for_shap_raw = X_raw.sample(n=min(1000, len(X_raw)), random_state=42)
    
    # Apply the preprocessor directly to the sample data to get the numerical features
    X_sample_processed = model_pipeline.named_steps['preprocessor'].transform(X_sample_for_shap_raw)

    classifier = model_pipeline.named_steps['classifier']
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_sample_processed)
    shap_values_class_1 = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig_shap_summary, ax_shap_summary = plt.subplots(figsize=(12, 10))
    if len(all_transformed_features) == shap_values_class_1.shape[1]:
        shap.summary_plot(shap_values_class_1, X_sample_processed, feature_names=all_transformed_features, show=False, ax=ax_shap_summary)
        ax_shap_summary.set_title('SHAP Feature Importance for Predicting Non-Adherence')
    else:
        shap.summary_plot(shap_values_class_1, X_sample_processed, show=False, ax=ax_shap_summary)
        ax_shap_summary.set_title('SHAP Feature Importance (Generic)')
        print("WARNING: Could not perfectly align SHAP values with feature names.")
    
    plt.tight_layout()
    return fig_shap_summary


def get_financial_impact_content():
    if df is None:
        return gr.Markdown("## Error: Data not loaded. Cannot display Financial Impact.")

    # Assumptions (same as in Section 3)
    average_drug_price_per_month = 500 # Simulated average drug price in INR/USD per month
    missed_months_if_dropout = 6 # Average number of months a patient might miss if they drop out

    at_risk_patients_for_finance = df[df['Predicted_Adherent'] == 1].copy()

    if at_risk_patients_for_finance.empty:
        return gr.Markdown("No patients predicted as 'Not Adherent' to estimate financial impact.")

    at_risk_patients_for_finance['Potential_Revenue_Loss'] = (
        at_risk_patients_for_finance['Dropout_Risk_Probability'] *
        average_drug_price_per_month *
        missed_months_if_dropout
    )

    total_estimated_revenue_loss = at_risk_patients_for_finance['Potential_Revenue_Loss'].sum()

    impact_markdown = f"""
    ## Estimated Financial Impact of Non-Adherence
    | Metric                                   | Value                   |
    | :--------------------------------------- | :---------------------- |
    | **Total Estimated Revenue Loss** | **₹{total_estimated_revenue_loss:,.2f}** |
    | *(Assumes avg drug price of ₹{average_drug_price_per_month} for {missed_months_if_dropout} missed months)* | |
    """

    # Revenue Loss Breakdown by Age Group
    revenue_loss_by_age = at_risk_patients_for_finance.groupby('Age_Group')['Potential_Revenue_Loss'].sum().sort_values(ascending=False)
    fig_rev_age, ax_rev_age = plt.subplots(figsize=(10, 6))
    sns.barplot(x=revenue_loss_by_age.index, y=revenue_loss_by_age.values, palette='RdYlGn_r', ax=ax_rev_age)
    ax_rev_age.set_title('Estimated Revenue Loss by Age Group')
    ax_rev_age.set_xlabel('Age Group')
    ax_rev_age.set_ylabel('Estimated Revenue Loss (₹)')
    ax_rev_age.ticklabel_format(style='plain', axis='y')
    ax_rev_age.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    fig_rev_cluster = None
    if 'Behavioral_Cluster' in df.columns: # Check if clustering was performed and column exists
        revenue_loss_by_cluster = at_risk_patients_for_finance.groupby('Behavioral_Cluster')['Potential_Revenue_Loss'].sum().sort_values(ascending=False)
        fig_rev_cluster, ax_rev_cluster = plt.subplots(figsize=(8, 6))
        sns.barplot(x=revenue_loss_by_cluster.index, y=revenue_loss_by_cluster.values, palette='viridis', ax=ax_rev_cluster)
        ax_rev_cluster.set_title('Estimated Revenue Loss by Behavioral Segment')
        ax_rev_cluster.set_xlabel('Behavioral Segment (Cluster)')
        ax_rev_cluster.set_ylabel('Estimated Revenue Loss (₹)')
        ax_rev_cluster.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
    
    return impact_markdown, fig_rev_age, fig_rev_cluster


def chat_logic(message, history):
    if df is None:
        return "Error: Data not loaded. Cannot respond."

    message = message.lower()
    response = "I'm sorry, I can only answer specific questions about overall risk, or risk by gender/age group/count of risks. Please refer to the examples below."

    if "overall dropout risk" in message or "total non-adherence" in message:
        predicted_non_adherent_count = df['Predicted_Adherent'].sum()
        total_patients = len(df)
        percentage = df['Predicted_Adherent'].mean() * 100
        response = f"Overall, **{predicted_non_adherent_count} out of {total_patients} patients ({percentage:.2f}%)** are predicted to be Not Adherent (at-risk)."

    elif "risk for" in message and "age group" in message and ("male" in message or "female" in message):
        gender_query = 'Male' if 'male' in message else 'Female'
        age_group_found = None
        for ag in df['Age_Group'].unique():
            if ag.lower().replace('-', '').replace('+', '') in message.replace('-', '').replace('+', ''):
                age_group_found = ag
                break

        if age_group_found:
            filtered_df = df[(df['Gender'] == gender_query) & (df['Age_Group'] == age_group_found)]
            if not filtered_df.empty:
                risk_percentage = filtered_df['Predicted_Adherent'].mean() * 100
                response = f"For **{gender_query} patients in Age Group {age_group_found}**, the predicted non-adherence risk is **{risk_percentage:.2f}%**."
            else:
                response = f"No {gender_query} patients found in Age Group {age_group_found}."
        else:
            response = "Please specify a valid age group (e.g., '35-45', '75+')."

    elif "highest risk" in message and ("age" in message or "age group" in message):
        risk_by_age = df.groupby('Age_Group')['Predicted_Adherent'].mean().sort_values(ascending=False)
        highest_risk_age_group = risk_by_age.index[0]
        highest_risk_percentage = risk_by_age.values[0] * 100
        response = f"The **Age Group with the highest predicted non-adherence risk** is **{highest_risk_age_group}** at **{highest_risk_percentage:.2f}%**."

    elif "highest risk" in message and "gender" in message:
        risk_by_gender = df.groupby('Gender')['Predicted_Adherent'].mean().sort_values(ascending=False)
        highest_risk_gender = risk_by_gender.index[0]
        highest_risk_percentage = risk_by_gender.values[0] * 100
        response = f"The **Gender with the highest predicted non-adherence risk** is **{highest_risk_gender}** at **{highest_risk_percentage:.2f}%**."

    elif "risk for patients with" in message and "risks" in message:
        try:
            num_risks = int(''.join(filter(str.isdigit, message.split('risks')[0])))
            filtered_df = df[df['Count_Of_Risks'] == num_risks]
            if not filtered_df.empty:
                risk_percentage = filtered_df['Predicted_Adherent'].mean() * 100
                response = f"For patients with **{num_risks} risks**, the predicted non-adherence risk is **{risk_percentage:.2f}%**."
            else:
                response = f"No patients found with {num_risks} risks in the dataset."
        except ValueError:
            response = "Please specify a valid number of risks (e.g., 'Show risk for patients with 2 risks.')."
    
    return response

# --- Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft(), title="RxRetention AI Agent") as demo:
    gr.Markdown("# 💊 RxRetention AI: Smart Agent for Pharma Adherence")
    gr.Markdown("## Predicting prescription dropout and estimating financial impact to enable targeted interventions.")

    with gr.Tab("Overview"):
        overview_metrics = gr.Markdown()
        overview_plot = gr.Plot()
        gr.Markdown("Interpretation: High non-adherence early in treatment emphasizes the need for early interventions.")
        # Call the content function on tab select or at load
        demo.load(get_overview_content, outputs=[overview_metrics, overview_plot])

    with gr.Tab("Adherence Insights"):
        gr.Markdown("## Deep Dive into Adherence Trends")
        adherence_plot_age = gr.Plot()
        adherence_plot_gender = gr.Plot()
        adherence_plot_risk = gr.Plot()
        # Call the content function on tab select or at load
        demo.load(get_adherence_insights_content, outputs=[adherence_plot_age, adherence_plot_gender, adherence_plot_risk])

    with gr.Tab("Model Explainability (SHAP)"):
        gr.Markdown("## Understanding Model Predictions with SHAP")
        gr.Markdown("SHAP (SHapley Additive exPlanations) helps explain how each feature contributes to the model's output (positive contribution means higher likelihood of non-adherence).")
        shap_plot = gr.Plot()
        # Call the content function on tab select or at load
        demo.load(get_shap_plot_content, outputs=shap_plot)

    with gr.Tab("Financial Impact"):
        financial_metrics = gr.Markdown()
        financial_plot_age = gr.Plot()
        financial_plot_cluster = gr.Plot()
        # Call the content function on tab select or at load
        demo.load(get_financial_impact_content, outputs=[financial_metrics, financial_plot_age, financial_plot_cluster])

    with gr.Tab("Chat with AI Agent"):
        gr.Markdown("## Chat with the RxRetention AI Agent")
        gr.Markdown("Ask questions about overall risk, or risk for specific demographic groups.")
        gr.Examples(
            examples=[
                "Show overall dropout risk",
                "What is the risk for Female patients in Age Group 75+?",
                "Which age group has the highest risk?",
                "Show risk for patients with 3 risks."
            ],
            inputs=gr.Textbox(label="Example Queries", placeholder="Example Queries")
        )
        gr.ChatInterface(chat_logic, title="Ask Your Agent")

demo.launch()