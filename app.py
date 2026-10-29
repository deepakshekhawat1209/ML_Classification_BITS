import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Telco Customer Churn Classification", layout="wide")

st.title("Telco Customer Churn Classification - ML Models")
st.write("Upload your test data CSV (with encoded feature columns + target column) to evaluate saved models.")


@st.cache_resource
def load_artifacts():
    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "SVM": "svm.joblib"
    }
    models = {}
    for name, fname in model_files.items():
        models[name] = joblib.load(os.path.join("model", fname))
    scaler = joblib.load(os.path.join("model", "scaler.joblib"))
    # exact feature columns (and their order) used during training
    feature_names = list(joblib.load(os.path.join("model", "feature_names.joblib")))
    return models, scaler, feature_names


models, scaler, feature_names = load_artifacts()

uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

    # Validate the uploaded CSV before using it:
    # it must have the target column and every feature the models were trained on
    has_target = "target" in test_data.columns
    missing_features = [c for c in feature_names if c not in test_data.columns]
    extra_cols = [c for c in test_data.columns if c not in feature_names and c != "target"]
    bad_target = has_target and not set(test_data["target"].dropna().unique()).issubset({0, 1})

    if not has_target or missing_features or bad_target:
        if not has_target:
            st.error("CSV must contain a target column (0 = No Churn, 1 = Churn)")
        if bad_target:
            st.error("The target column must contain only 0 and 1")
        if missing_features:
            st.error(f"CSV is missing {len(missing_features)} required feature column(s): {', '.join(missing_features)}")
        st.info(f"Expected {len(feature_names)} feature columns + target. Use test_data.csv from the repo as a template.")
    elif [c for c in feature_names if not pd.api.types.is_numeric_dtype(test_data[c])]:
        non_numeric = [c for c in feature_names if not pd.api.types.is_numeric_dtype(test_data[c])]
        st.error(f"These feature columns must be numeric (already encoded): {', '.join(non_numeric)}")
    elif test_data[feature_names + ["target"]].isnull().any().any():
        st.error("Some values are empty - please fill or remove those rows before uploading")
    else:
        if extra_cols:
            st.warning(f"Ignoring {len(extra_cols)} extra column(s) not used by the models: {', '.join(extra_cols)}")

        # Keep only the training features, in the same order the scaler and models saw in training
        X_test = test_data[feature_names]
        y_test = test_data["target"].astype(int)
        st.success(f"CSV validated: {len(test_data)} rows, {len(feature_names)} features + target")

        # Quick look at the uploaded file so we can confirm it was read correctly
        st.subheader("Uploaded Test Data (first 5 rows)")
        st.write(f"{test_data.shape[0]} rows x {test_data.shape[1]} columns")
        st.dataframe(test_data.head(), use_container_width=True)

        # Apply the same scaling that was used during training (keep column names, the models were fit with them)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

        # Model selection dropdown
        st.subheader("Select Model")
        selected = st.selectbox("Choose a model to evaluate", list(models.keys()))

        model = models[selected]
        y_pred = model.predict(X_test_scaled)

        # AUC needs probability scores (all 5 saved models support predict_proba)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        # Calculate all 6 metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Display evaluation metrics
        st.subheader("Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col1.metric("Precision", f"{precision:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col2.metric("Recall", f"{recall:.4f}")
        col3.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {selected}")
        st.pyplot(fig)

        # Classification report
        st.subheader("Classification Report")
        # output_dict=True gives the report as a dict, so it can be shown as a proper table
        report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        # the accuracy row has only one real value, so clear the repeated ones and show the row count as support
        report_df.loc["accuracy", ["precision", "recall"]] = None
        report_df.loc["accuracy", "support"] = len(y_test)
        report_df["support"] = report_df["support"].astype(int)
        report_df.columns = ["Precision", "Recall", "F1-score", "Support"]
        st.dataframe(report_df.round(4), use_container_width=True)

        # All models comparison table
        st.subheader("All Models Comparison")
        all_results = []
        for name, m in models.items():
            pred = m.predict(X_test_scaled)
            prob = m.predict_proba(X_test_scaled)[:, 1]
            a = roc_auc_score(y_test, prob)
            all_results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "AUC": a,
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1": f1_score(y_test, pred),
                "MCC": matthews_corrcoef(y_test, pred)
            })
        comp_df = pd.DataFrame(all_results)
        st.dataframe(comp_df.round(4), use_container_width=True)

        # Short summary built from the comparison table, so it always matches the uploaded data
        best = comp_df.loc[comp_df["MCC"].idxmax()]
        top_recall = comp_df.loc[comp_df["Recall"].idxmax()]
        churn_pct = y_test.mean() * 100
        st.subheader("Summary")
        st.markdown(f"- **Best overall model:** {best['Model']}, with the highest MCC ({best['MCC']:.4f}) and an F1 of {best['F1']:.4f} on this test data.")
        st.markdown(f"- **Catches the most churners:** {top_recall['Model']}, with a recall of {top_recall['Recall']:.4f}, useful if missing a churner costs more than a false alarm.")
        st.markdown(f"- **Why MCC decides the winner:** only {churn_pct:.1f}% of customers in this file churned, so accuracy can look good even when many churners are missed. MCC uses all four cells of the confusion matrix.")
else:
    st.info("Please upload a CSV file with the test data.")
