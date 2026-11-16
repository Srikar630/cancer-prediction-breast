import joblib
import numpy as np
import streamlit as st

bundle = joblib.load("model.pkl")
model = bundle["model"]
features = bundle["features"]
means, mins, maxs = bundle["means"], bundle["mins"], bundle["maxs"]

st.title("🩺 Breast Cancer Predictor")
st.caption("Educational demo only, not medical advice.")

# Show 10 'mean' features as sliders; the rest use dataset averages
shown = features[:10]
values = list(means)
cols = st.columns(2)
for i, name in enumerate(shown):
    with cols[i % 2]:
        values[i] = st.slider(
            name, float(mins[i]), float(maxs[i]), float(means[i])
        )

if st.button("Predict"):
    x = np.array(values).reshape(1, -1)
    prob_benign = model.predict_proba(x)[0][1]
    if prob_benign >= 0.5:
        st.success(f"Benign ({prob_benign:.1%} confidence)")
    else:
        st.error(f"Malignant ({1 - prob_benign:.1%} confidence)")
