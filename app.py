import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🔍 Root Cause Analysis App")

uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    target_column = st.selectbox("Select the Result Column (Target)", all_columns)
    variable_columns = st.multiselect("Select Variable Columns (Causes)", [col for col in all_columns if col != target_column])

    if st.button("Analyze"):
        if target_column and variable_columns:
            df = df.dropna()
            X = df[variable_columns]
            y = df[target_column]

            if y.nunique() > 10:
                st.warning("Target variable has too many unique values. Consider using a simplified or categorical column.")
            else:
                # Encode non-numeric columns
                X = pd.get_dummies(X)
                if y.dtype == 'object':
                    y = pd.factorize(y)[0]

                model = DecisionTreeClassifier(max_depth=4)
                model.fit(X, y)

                importance = pd.Series(model.feature_importances_, index=X.columns)
                sorted_importance = importance.sort_values(ascending=False)

                st.subheader("🔑 Top Root Cause Variables:")
                st.write(sorted_importance.head())

                fig = plt.figure(figsize=(10, 4))
                sns.barplot(x=sorted_importance.values, y=sorted_importance.index)
                plt.title("Feature Importance")
                st.pyplot(fig)

                st.subheader("📄 Model Summary:")
                y_pred = model.predict(X)
                st.text(classification_report(y, y_pred))
        else:
            st.warning("Please select both target and variable columns.")
