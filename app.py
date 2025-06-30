import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import os

# Load OpenAI Key from environment
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🔍 Smart Root Cause Analyzer (with GPT)")

uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])

def explain_root_causes(variable_importance, df, target_column):
    try:
        top_vars = variable_importance.head(3).index.tolist()
        sample = df[top_vars + [target_column]].dropna().head(20).to_dict(orient='records')
        prompt = f"""
        You are a business analyst helping users understand root causes behind changes in a result column (“{target_column}”).
        Based on these top 3 variables and sample data rows, provide a clear, practical explanation of what's most likely driving changes.

        Top Variables: {top_vars}
        Sample Data: {sample}

        Give a human-friendly summary in plain English with actionable recommendations.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful and analytical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=300
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ Could not generate GPT explanation: {e}"

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Dataset Preview:")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    target_column = st.selectbox("🎯 Select Target Column (Result)", all_columns)
    variable_columns = st.multiselect("📊 Select Variable Columns (Possible Causes)", [col for col in all_columns if col != target_column])

    if st.button("Analyze Root Causes"):
        if target_column and variable_columns:
            df = df.dropna()
            X = pd.get_dummies(df[variable_columns])
            y = df[target_column]

            is_categorical = y.dtype == 'object' or y.nunique() < 10

            if is_categorical:
                st.info("🔍 Using Classification (Target is Categorical)")
                y_encoded, classes = pd.factorize(y)
                model = DecisionTreeClassifier(max_depth=4)
                model.fit(X, y_encoded)
                y_pred = model.predict(X)

                importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                st.subheader("🔑 Top Influencing Variables")
                st.write(importance.head())

                fig = plt.figure(figsize=(10, 4))
                sns.barplot(x=importance.values, y=importance.index)
                plt.title("Feature Importance")
                st.pyplot(fig)

                st.subheader("📈 Classification Report")
                st.text(classification_report(y_encoded, y_pred, target_names=classes))

            else:
                st.info("📈 Using Regression (Target is Continuous)")
                model = DecisionTreeRegressor(max_depth=4)
                model.fit(X, y)
                y_pred = model.predict(X)

                importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                st.subheader("🔑 Top Influencing Variables")
                st.write(importance.head())

                fig = plt.figure(figsize=(10, 4))
                sns.barplot(x=importance.values, y=importance.index)
                plt.title("Feature Importance")
                st.pyplot(fig)

                st.subheader("📊 Regression Performance")
                mae = mean_absolute_error(y, y_pred)
                rmse = mean_squared_error(y, y_pred, squared=False)
                st.write(f"**MAE:** {mae:.2f}")
                st.write(f"**RMSE:** {rmse:.2f}")

                st.subheader("🧪 Sample Predictions")
                st.write(pd.DataFrame({"Actual": y, "Predicted": y_pred}).head())

            st.subheader("🧠 AI Explanation of Root Causes")
            explanation = explain_root_causes(importance, df[variable_columns + [target_column]], target_column)
            st.markdown(explanation)
        else:
            st.warning("Please select both a target and at least one variable.")
