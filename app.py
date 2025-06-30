import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🔍 Root Cause Analysis App (Smart Detection)")

uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    target_column = st.selectbox("Select the Target Column", all_columns)
    variable_columns = st.multiselect("Select Variable Columns (Causes)", [col for col in all_columns if col != target_column])

    if st.button("Analyze"):
        if target_column and variable_columns:
            df = df.dropna()
            X = df[variable_columns]
            y = df[target_column]

            # Encode categorical variables
            X = pd.get_dummies(X)

            # Auto-detect: Classification vs Regression
            use_classification = y.dtype == 'object' or y.nunique() < 10

            if use_classification:
                st.info("🔍 Using Classification Model (Decision Tree Classifier)")
                y_encoded, classes = pd.factorize(y)
                model = DecisionTreeClassifier(max_depth=4)
                model.fit(X, y_encoded)
                y_pred = model.predict(X)

                importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

                st.subheader("🔑 Top Influencing Variables:")
                st.write(importance.head())

                fig = plt.figure(figsize=(10, 4))
                sns.barplot(x=importance.values, y=importance.index)
                plt.title("Feature Importance")
                st.pyplot(fig)

                st.subheader("📊 Classification Report:")
                st.text(classification_report(y_encoded, y_pred, target_names=classes))

            else:
                st.info("📈 Using Regression Model (Decision Tree Regressor)")
                model = DecisionTreeRegressor(max_depth=4)
                model.fit(X, y)
                y_pred = model.predict(X)

                importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

                st.subheader("🔑 Top Influencing Variables:")
                st.write(importance.head())

                fig = plt.figure(figsize=(10, 4))
                sns.barplot(x=importance.values, y=importance.index)
                plt.title("Feature Importance")
                st.pyplot(fig)

                st.subheader("📈 Regression Performance:")
                mae = mean_absolute_error(y, y_pred)
                rmse = mean_squared_error(y, y_pred, squared=False)
                st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
                st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")

                st.subheader("🔍 Sample Predictions:")
                st.write(pd.DataFrame({"Actual": y, "Predicted": y_pred}).head())
        else:
            st.warning("Please select both target and variable columns.")
