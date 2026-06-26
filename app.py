import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return joblib.load("crop_recommendation_model.pkl")

model_package = load_model()
model = model_package["model"]
features = model_package["features"]
classes = model_package["classes"]

st.set_page_config(page_title="AgriCrop AI", page_icon="🌱", layout="wide")

st.title("🌱 AgriCrop AI")
st.subheader("Smart Crop Recommendation System")
st.write("This web app predicts the most suitable crop based on soil nutrients and climate conditions using a Random Forest classification model.")
st.warning("This is a decision-support tool for a class project. Real farm decisions should also consider crop variety, soil texture, irrigation, pests, diseases, location, and market demand.")
st.divider()

st.sidebar.header("Model Information")
st.sidebar.write("Algorithm: Random Forest Classifier")
st.sidebar.write("Task: Multi-class crop classification")
st.sidebar.write("Inputs: N, P, K, temperature, humidity, pH, rainfall")
st.sidebar.write("Output: Recommended crop")

st.header("Single Crop Prediction")
st.write("Enter soil nutrient and climate values below.")

col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50, step=1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
with col2:
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50, step=1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
with col3:
    K = st.number_input("Potassium (K)", min_value=0, max_value=250, value=50, step=1)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

input_data = pd.DataFrame({
    "N": [N],
    "P": [P],
    "K": [K],
    "temperature": [temperature],
    "humidity": [humidity],
    "ph": [ph],
    "rainfall": [rainfall]
})

st.subheader("Input Data Preview")
st.dataframe(input_data, use_container_width=True)

if st.button("Predict Suitable Crop"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = probabilities.max() * 100

    st.success(f"Recommended Crop: **{prediction.upper()}**")
    st.info(f"Model Confidence: **{confidence:.2f}%**")

    probability_df = pd.DataFrame({
        "Crop": classes,
        "Probability (%)": probabilities * 100
    }).sort_values(by="Probability (%)", ascending=False).head(5)

    st.subheader("Top 5 Crop Recommendations")
    st.dataframe(probability_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(probability_df["Crop"], probability_df["Probability (%)"])
    ax.set_xlabel("Crop")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Top 5 Crop Recommendation Probability")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

st.divider()

st.header("Batch Prediction")
st.write("Upload a CSV file with these columns: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`.")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(batch_df, use_container_width=True)

    missing_columns = [col for col in features if col not in batch_df.columns]
    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
    else:
        batch_predictions = model.predict(batch_df[features])
        batch_confidence = model.predict_proba(batch_df[features]).max(axis=1) * 100
        result_df = batch_df.copy()
        result_df["Recommended Crop"] = batch_predictions
        result_df["Confidence (%)"] = batch_confidence.round(2)

        st.success("Batch prediction completed.")
        st.subheader("Prediction Results")
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="crop_recommendation_results.csv",
            mime="text/csv"
        )

st.divider()
st.header("About This Model")
st.write("""
This crop recommendation system uses a Random Forest Classifier.

Input variables:
- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- Soil pH
- Rainfall

Output:
- Recommended crop
- Confidence score
- Top 5 crop suggestions

The model is useful for demonstrating AI-based decision support in agriculture. However, the dataset is clean and balanced, so real-world field performance may be lower.
""")
