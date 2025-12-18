# import streamlit as st
# import joblib
# import numpy as np

# bundle = joblib.load("knn_model1.pkl")
# model = bundle["model"]
# best_cols = bundle["best_cols"]
# label_map = bundle["label_map"]

# st.title("Breast Cancer KNN Predict")

# inputs = []
# for col in best_cols:
#     val = st.number_input(col, value=0.0, format="%.6f")
#     inputs.append(val)

# if st.button("Predict"):
#     X = np.array(inputs, dtype=float).reshape(1, -1)
#     # X_static = np.array([
#     #     [1.789e-01, 1.659e+02, 7.951e-02, 2.486e+01, 1.236e+02,
#     #      1.866e+03, 1.894e+01, 1.130e+03, 1.080e-01, 2.687e-01]
#     # ], dtype=float)
#     pred = int(model.predict(X)[0])
#     proba = model.predict_proba(X)[0]

#     inv_label = {v: k for k, v in label_map.items()}

#     st.write("X:", X)
#     st.write("pred:", pred)
#     st.write("proba:", proba)
#     st.success(f"Natija: {inv_label[pred]}  (B=Sog'lom, M=Kasal)")
#     st.write(f"Probabilities: B={proba[0]:.4f}, M={proba[1]:.4f}")




import streamlit as st
import joblib
import numpy as np

bundle = joblib.load("knn_model1.pkl")
model = bundle["model"]
best_cols = bundle["best_cols"]
label_map = bundle["label_map"]

st.title("Breast Cancer KNN Predict")

inputs = []
for col in best_cols:
    val = st.number_input(col, value=0.0, format="%.6f")
    inputs.append(val)

if st.button("Predict"):
    X = np.array(inputs, dtype=float).reshape(1, -1)

    # X_static = np.array([
    #     [1.789e-01, 1.659e+02, 7.951e-02, 2.486e+01, 1.236e+02,
    #      1.866e+03, 1.894e+01, 1.130e+03, 1.080e-01, 2.687e-01]
    # ], dtype=float)

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    st.write("X_static:", X)
    st.write("Raw pred value:", pred)
    st.write("Probabilities:", proba)

    if pred == 0:
        st.success("✅ Natija: SOG'LOM (SALBIY / NEGATIVE)")
    else:
        st.error("❌ Natija: KASAL (IJOBIY / POSITIVE)")

    st.write(f"Ishonchlilik: Sog'lom={proba[0]:.2%}, Kasal={proba[1]:.2%}")
