import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import os
import numpy as np


def load_openai_api_key() -> str:
    """Retrieve the OpenAI API key from env vars or Streamlit secrets."""
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    return st.secrets.get("OPENAI_API_KEY", "")


def main() -> None:
    """Run the Streamlit application."""
    openai.api_key = load_openai_api_key()

    st.title("🔍 Smart Root Cause Analyzer (with GPT)")

    uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])

# GPT-based explanation function
def explain_root_causes(variable_importance, df, target_column, original_columns_map):
    try:
        top_dummy_vars = variable_importance.head(3).index.tolist()

        # Map dummy back to original column names
        top_original_vars = []
        for dummy_col in top_dummy_vars:
            for orig_col, dummy_list in original_columns_map.items():
                if dummy_col in dummy_list:
                    top_original_vars.append(orig_col)
                    break

        top_original_vars = list(set(top_original_vars))[:3]
        sample = df[top_original_vars + [target_column]].dropna().head(20).to_dict(orient='records')

        prompt = f"""
        You are a business analyst helping users understand root causes behind changes in the result column (“{target_column}”).
        Based on these top 3 variables and sample data rows, provide a clear, practical explanation of what is most likely driving changes.

        Top Variables: {top_original_vars}
        Sample Data: {sample}

        Give a plain-English, actionable root cause explanation.
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


    # Main app logic
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
                X = df[variable_columns]
                y = df[target_column]

                # Encode categorical variables
                X = pd.get_dummies(X)

                # Create mapping for original vs dummy columns
                original_columns_map = {
                    col: [c for c in X.columns if c.startswith(col + "_") or c == col]
                    for col in variable_columns
                }

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
                    try:
                        st.text(classification_report(y_encoded, y_pred, target_names=[str(c) for c in classes]))
                    except Exception as e:
                        st.warning(f"⚠️ Could not display classification report: {e}")

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
                    mse = mean_squared_error(y, y_pred)
                    rmse = np.sqrt(mse)

                    st.write(f"**MAE:** {mae:.2f}")
                    st.write(f"**RMSE:** {rmse:.2f}")

                    st.subheader("🧪 Sample Predictions")
                    st.write(pd.DataFrame({"Actual": y, "Predicted": y_pred}).head())

                # GPT Explanation Section
                st.subheader("🧠 AI Explanation of Root Causes")
                explanation = explain_root_causes(
                    importance,
                    df[variable_columns + [target_column]],
                    target_column,
                    original_columns_map
                )
                st.markdown(explanation)

            else:
                st.warning("Please select both a target and at least one variable.")


if __name__ == "__main__":
    main()
