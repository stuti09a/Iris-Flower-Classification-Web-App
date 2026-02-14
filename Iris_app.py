# Iris_app.py - Streamlit App for Iris model prediction and basic exploration

# Importing Libraries
# -------------------------------
# Iris Classification Demo App
# -------------------------------

# -------------------------------------------
# Iris Flower Classification - Streamlit App
# -------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and dataset
model = joblib.load("iris_model.joblib")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Streamlit Page Configuration

st.set_page_config(page_title="Iris Classifier Dashboard 🌸", layout="wide")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Choose a mode:", ["Data Exploration", "Prediction"])

st.sidebar.info("Switch between exploring the dataset and predicting flower species.")

# Data Exploration

if page == "Data Exploration":
    st.title("Iris Dataset Exploration Dashboard")
    st.write("Explore patterns and relationships in the Iris dataset used to train the model.")

    with st.container():
        st.subheader("🔹 Dataset Overview")
        st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1: # Histogram
        st.subheader("📈 Feature Distribution (Histogram)")
        feature = st.selectbox("Select a feature:", iris.feature_names)
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, color="cornflowerblue", ax=ax)
        ax.set_xlabel(feature)
        st.pyplot(fig)

    with col2:  # Scatter Plot
        st.subheader("🌿 Scatter Plot of Feature Pairs")
        x_axis = st.selectbox("X-axis:", iris.feature_names, index=0)
        y_axis = st.selectbox("Y-axis:", iris.feature_names, index=1)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue="species", palette="Set2", s=80)
        ax.set_title(f"{x_axis} vs {y_axis}")
        st.pyplot(fig)

    st.caption("Insights: Setosa species typically have smaller petals, while Virginica shows larger feature ranges.")

# Flower Species Prediction
elif page == "Prediction":
    st.title("🌸 Iris Flower Species Prediction")
    st.write("""
    Enter your flower measurements below to predict flower species.
    The model is trained using Random Forest Classifier on the classic Iris dataset.
    """)

    # Layout for sliders
    with st.container():
        col1, col2 = st.columns(2)

        with col1: # Sepal Details
            sepal_length = st.slider(
                "Sepal Length (cm)",
                4.0, 8.0, 5.8,
                help="Length of the outer part of the flower (in cm)."
            )
            sepal_width = st.slider(
                "Sepal Width (cm)",
                2.0, 4.5, 3.0,
                help="Width of the outer part of the flower (in cm)."
            )

        with col2:  # Petal Details
            petal_length = st.slider(
                "Petal Length (cm)",
                1.0, 7.0, 4.2,
                help="Length of the inner petal (in cm)."
            )
            petal_width = st.slider(
                "Petal Width (cm)",
                0.1, 2.5, 1.2,
                help="Width of the inner petal (in cm)."
            )

    # Prediction button
    if st.button("Predict Flower Species"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)
        species = ['Setosa', 'Versicolor', 'Virginica']
        predicted_species = species[prediction[0]]
        confidence = np.max(prediction_proba[0]) * 100

        # Conditional color formatting
        if confidence > 85:
            color = "🟢"    # High Confidence
            message = "High confidence prediction"
        elif confidence > 60:
            color = "🟡"    # Moderate Confidence
            message = "Moderate confidence prediction"
        else:
            color = "🔴"    # Low Confidence
            message = "Low confidence prediction"

        st.markdown("---")
        st.subheader(f"{color} Predicted Species: **{predicted_species}**")
        st.write(f"**Confidence Level:** {confidence:.2f}% ({message})")

        st.progress(int(confidence))

        with st.expander("View Prediction Probabilities"):
            prob_df = pd.DataFrame(
                prediction_proba,
                columns=species
            ).T.rename(columns={0: "Probability"})
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

    st.caption("Tip: Adjust the sliders to see how sepal/petal changes affect predictions.")

# Footer

st.markdown("---")
st.caption("Developed by Stuti Agarwal | Random Forest Classifier | Streamlit APP Demo")