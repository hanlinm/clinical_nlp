import os
import json
import pickle
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from evaluator import evaluate_single_prediction

load_dotenv()

st.set_page_config(
    page_title="Clinical NLP Classifier",
    page_icon="🏥",
    layout="wide"
)

# Load the trained classifier
@st.cache_resource
def load_classifier():
    if not os.path.exists("models/classifier.pkl"):
        with st.status("Training classifier for first deployment..."):
            from classifier import build_and_evaluate
            os.makedirs("models", exist_ok=True)
            build_and_evaluate()
    with open("models/classifier.pkl", "rb") as f:
        return pickle.load(f)

# Load evaluation results
@st.cache_data
def load_eval_results():
    if not os.path.exists("data/evaluation_result.csv"):
        with st.status("Running LLM evaluation for first deployment... (2-3 mins)"):
            from evaluator import evaluate_test_set
            evaluate_test_set()
    if not os.path.exists("data/evaluation_result.csv"):
        return None
    return pd.read_csv("data/evaluation_result.csv")
if eval_df is None:
    st.error("Evaluation results not found. Please run `python evaluator.py` first.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🏥 Clinical NLP Classifier")
    st.markdown("""
    This app classifies clinical notes into medical specialties
    using a TF-IDF + Logistic Regression classifier, then uses
    GPT-4o-mini to evaluate and explain the prediction.
    """)
    st.divider()
    st.markdown("**Supported specialties:**")
    st.markdown("""
    - Surgery
    - Cardiovascular / Pulmonary
    - Orthopedic
    - Radiology
    - Neurology
    """)
    st.divider()
    st.caption("Built with scikit-learn, OpenAI, and Streamlit")

# ── Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Classify a Note", "📊 Evaluation Dashboard"])

# ── Tab 1: Single note classifier ────────────────────────────────────────
with tab1:
    st.header("Classify a Clinical Note")
    st.caption("Paste a clinical note below to get a specialty prediction and LLM evaluation")

    # Example notes to make the demo easy to use
    examples = {
        "Select an example...": "",
        "Orthopedic — knee surgery": """The patient is a 58-year-old male presenting with 
chronic right knee pain and instability. MRI revealed a complete tear of the anterior 
cruciate ligament with associated medial meniscus tear. The patient underwent arthroscopic 
ACL reconstruction using a patellar tendon autograft. Postoperative course was unremarkable. 
Physical therapy initiated on day 2.""",
        "Cardiology — chest pain": """Patient is a 67-year-old female presenting with 
acute onset chest pain radiating to the left arm, diaphoresis, and shortness of breath. 
EKG showed ST elevation in leads II, III, and aVF. Troponin elevated at 2.4. 
Emergent cardiac catheterization performed revealing 95% occlusion of the right 
coronary artery. Drug-eluting stent placed successfully.""",
        "Neurology — seizure": """A 34-year-old male with no significant past medical 
history presented following a witnessed generalized tonic-clonic seizure lasting 
approximately 2 minutes. Post-ictal confusion noted for 20 minutes. MRI brain 
unremarkable. EEG showed no epileptogenic activity. Started on levetiracetam 500mg 
twice daily. Neurology follow-up scheduled in 4 weeks.""",
    }

    selected = st.selectbox("Try an example or paste your own below:", examples.keys())
    
    note_input = st.text_area(
        "Clinical note:",
        value=examples[selected],
        height=200,
        placeholder="Paste a clinical note here..."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        classify_btn = st.button("🔍 Classify + Evaluate", type="primary", use_container_width=True)
    with col2:
        st.caption("⚠️ LLM evaluation takes 5–10 seconds and uses a small amount of OpenAI credits")

    if classify_btn and note_input.strip():
        # Step 1: classifier prediction
        with st.spinner("Running classifier..."):
            prediction = classifier.predict([note_input])[0]
            probabilities = classifier.predict_proba([note_input])[0]
            classes = classifier.classes_

        st.subheader("Classifier Prediction")
        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            st.metric("Predicted Specialty", prediction)

        with pred_col2:
            # Show confidence scores for all classes
            prob_df = pd.DataFrame({
                "Specialty": classes,
                "Confidence": probabilities
            }).sort_values("Confidence", ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        st.divider()

        # Step 2: LLM evaluation
        st.subheader("LLM Evaluation")
        st.caption("GPT-4o-mini evaluates the prediction and explains its reasoning")

        with st.spinner("Asking GPT-4o-mini to evaluate..."):
            # We don't know the true label for user input so we pass prediction as both
            eval_result = evaluate_single_prediction(
                text=note_input,
                true_label="Unknown",
                predicted_label=prediction
            )

        eval_col1, eval_col2 = st.columns(2)
        with eval_col1:
            st.metric(
                "LLM Confidence",
                f"{eval_result.get('confidence', 0):.0%}"
            )
        with eval_col2:
            st.metric(
                "LLM Suggested Label",
                eval_result.get("suggested_label", "N/A")
            )

        st.markdown("**Reasoning:**")
        st.info(eval_result.get("reasoning", "No reasoning available"))

        terms = eval_result.get("key_clinical_terms", [])
        if terms:
            st.markdown("**Key clinical terms identified:**")
            st.write(" · ".join([f"`{t}`" for t in terms]))

    elif classify_btn:
        st.warning("Please paste a clinical note before classifying.")

# ── Tab 2: Evaluation Dashboard ───────────────────────────────────────────
with tab2:
    st.header("Evaluation Dashboard")
    st.caption("LLM evaluation results across a 30-sample test set")

    # Summary metrics
    valid = eval_df[eval_df["is_correct"].notna()]
    classifier_acc = pd.read_csv("data/test_results.csv")
    classifier_acc = (classifier_acc["label"] == classifier_acc["predicted"]).mean()
    llm_agreement = valid["is_correct"].mean()
    avg_confidence = valid["confidence"].mean()

    # Boundary case calculation
    boundary_keywords = ["reasonable", "subspecialty", "both", "while", "although", "however"]
    wrong = eval_df[eval_df["is_correct"] == False].copy()
    wrong["is_boundary"] = wrong["reasoning"].str.lower().apply(
        lambda x: any(word in x for word in boundary_keywords)
    )
    boundary_count = wrong["is_boundary"].sum()
    adjusted_acc = (valid["is_correct"].sum() + boundary_count) / len(valid)

    # Display metrics in a row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Classifier Accuracy", f"{classifier_acc:.1%}")
    with m2:
        st.metric("LLM Agreement", f"{llm_agreement:.1%}")
    with m3:
        st.metric("Adjusted Accuracy", f"{adjusted_acc:.1%}")
    with m4:
        st.metric("Avg LLM Confidence", f"{avg_confidence:.2f}")

    st.divider()

    # Misclassification breakdown
    st.subheader("Misclassification Analysis")
    wrong_display = wrong[["true_label", "predicted_label", "reasoning", "is_boundary"]].copy()
    wrong_display.columns = ["True Label", "Predicted", "LLM Reasoning", "Boundary Case"]
    st.dataframe(wrong_display, use_container_width=True, hide_index=True)

    st.divider()

    # Correct predictions
    st.subheader("Correct Predictions")
    correct = eval_df[eval_df["is_correct"] == True][["true_label", "predicted_label", "confidence", "reasoning"]]
    correct.columns = ["True Label", "Predicted", "LLM Confidence", "LLM Reasoning"]
    st.dataframe(correct, use_container_width=True, hide_index=True)