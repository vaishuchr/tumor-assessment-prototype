import os
from pathlib import Path

import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt



# Page setup (sets expectations + makes the dashboard feel intentional)
st.set_page_config(
    page_title="Tumor Assessment (Prototype)",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Small utilities
def _first_existing_path(paths: list[str]) -> str | None:
    for p in paths:
        if p and Path(p).exists():
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Caching keeps the dashboard snappy: Streamlit reruns the script often,
    and re-loading the model every time feels slow and laggy.
    """
    model_path = _first_existing_path([
        "model/classifier.pkl",   
        "classifier.pkl",         
    ])
    cm_path = _first_existing_path([
        "model/conf_matrix.pkl",
        "conf_matrix.pkl",
    ])
    metrics_path = _first_existing_path([
        "model/metrics.txt",
        "metrics.txt",
    ])

    if not model_path:
        raise FileNotFoundError(
            "I couldn't find classifier.pkl. I looked for 'model/classifier.pkl' and './classifier.pkl'."
        )

    model = joblib.load(model_path)

    conf_matrix = None
    if cm_path:
        try:
            conf_matrix = joblib.load(cm_path)
        except Exception:
            conf_matrix = None

    metrics = {}
    if metrics_path:
        try:
            raw = Path(metrics_path).read_text(encoding="utf-8")
            for line in raw.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    metrics[k.strip()] = v.strip()
        except Exception:
            metrics = {}

    return model, conf_matrix, metrics


def safe_predict(model, x: np.ndarray):
    """
    Predict in a way that works for many sklearn-like models.

    Returns:
      pred_class, human_label, prob_for_pred (or None), proba_vector (or None)
    """
    pred_class = model.predict(x)[0]

    proba_vec = None
    prob_for_pred = None
    if hasattr(model, "predict_proba"):
        proba_vec = model.predict_proba(x)[0]
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if pred_class in classes:
                prob_for_pred = float(proba_vec[classes.index(pred_class)])
            else:
                prob_for_pred = float(np.max(proba_vec))
        else:
            prob_for_pred = float(np.max(proba_vec))

    # label mapping
    if hasattr(model, "classes_") and set(model.classes_) == {0, 1}:
        human_label = "Malignant" if int(pred_class) == 1 else "Benign"
    else:
        human_label = f"Class {pred_class}"

    return pred_class, human_label, prob_for_pred, proba_vec


def draw_confusion_matrix(conf_matrix):
    """
    Simple matplotlib confusion matrix (keeps dependencies light).
    """
    fig, ax = plt.subplots()
    ax.imshow(conf_matrix, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    classes = ["Benign", "Malignant"]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center")

    fig.tight_layout()
    return fig


# Input definitions + soft validation ranges
FEATURES = [
    ("Radius (mean)", "Average distance from the center to the perimeter."),
    ("Texture (mean)", "How much the pixel intensity varies (roughness in appearance)."),
    ("Perimeter (mean)", "Average length around the nucleus boundary."),
    ("Area (mean)", "Average area inside the nucleus boundary."),
    ("Smoothness (mean)", "How smooth the boundary is (local radius variation)."),
    ("Compactness (mean)", "A shape measure: (Perimeter² / Area) - 1.0"),
    ("Concavity (mean)", "How deep the concave portions of the contour are."),
    ("Concave Points (mean)", "How many concave sections appear in the contour."),
    ("Symmetry (mean)", "How similar the nucleus looks when reflected."),
    ("Fractal Dimension (mean)", "A roughness measure of the boundary (complexity)."),
]

# These are deliberately “soft” ranges used to catch obvious input mistakes.
# They’re not medical thresholds and they’re not universal.
SOFT_RANGES = {
    "Radius (mean)": (6.0, 30.0),
    "Texture (mean)": (9.0, 40.0),
    "Perimeter (mean)": (40.0, 200.0),
    "Area (mean)": (100.0, 2500.0),
    "Smoothness (mean)": (0.04, 0.18),
    "Compactness (mean)": (0.01, 0.40),
    "Concavity (mean)": (0.00, 0.60),
    "Concave Points (mean)": (0.00, 0.25),
    "Symmetry (mean)": (0.10, 0.35),
    "Fractal Dimension (mean)": (0.04, 0.12),
}


def validate_inputs(values: list[float]) -> dict:
    """
    Returns a dict like:
      {
        "fatal": [..],
        "warnings": [..],
        "notes": [..]
      }

    "fatal" = strongly suggests the input is not meaningful
    "warnings" = suspicious values (likely typos or wrong units)
    """
    result = {"fatal": [], "warnings": [], "notes": []}

    # Big red flag: all zeros (common when someone just clicks through)
    if all(v == 0.0 for v in values):
        result["fatal"].append(
            "All fields are 0. That usually means the inputs weren’t entered yet, so the output won’t be meaningful."
        )
        return result

    # Another flag: too many zeros (often “I don’t know these”)
    zero_count = sum(1 for v in values if v == 0.0)
    if zero_count >= 5:
        result["warnings"].append(
            f"{zero_count} fields are 0. If you’re unsure about a value, it’s better to use sample demo values than guess."
        )

    # Soft range checks (typos, wrong units, copy/paste errors)
    for (label, _), v in zip(FEATURES, values):
        lo, hi = SOFT_RANGES.get(label, (None, None))
        if lo is None:
            continue

        if v < lo or v > hi:
            result["warnings"].append(
                f"**{label}** looks unusual ({v:.4f}). Typical demo values are often around {lo}–{hi}. "
                "This might be totally fine, but it’s worth double-checking for typos or unit mix-ups."
            )

    # Gentle note for the user
    result["notes"].append(
        "These checks are just guardrails to catch obvious mistakes. They are not medical limits."
    )

    return result


# Load model artifacts
try:
    model, conf_matrix, metrics = load_artifacts()
except Exception as e:
    st.error("I couldn’t load the model files for this app.")
    st.exception(e)
    st.stop()


# Header + disclaimer
st.title("Breast Tumor Assessment (ML Prototype)")
st.caption(
    "This is a human-centered decision-support demo. It estimates risk based on numeric features—"
    "it doesn’t diagnose cancer and it doesn’t replace a clinician."
)

with st.expander("Medical disclaimer (please read)", expanded=True):
    st.markdown(
        """
**Educational / research demo only. Not a medical device.**  
This tool does **not** provide a diagnosis. It simply compares your inputs to patterns the model learned from training data.

It does **not** include:
- patient history,
- imaging review (mammogram/ultrasound/MRI),
- pathologist interpretation,
- lab confirmation,
- or clinical judgment.

If you have a real health concern, talk to a licensed healthcare professional.
        """
    )


# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Where do you want to go?",
    ["Assess", "How to Use", "Model & Performance", "Limitations & Ethics"],
)

st.sidebar.divider()
st.sidebar.markdown(
    """
**Quick tip:** If you’re just testing the UI, turn on **sample demo values** on the Assess page.
"""
)


# Page: Assess
if page == "Assess":
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("1) Enter feature values")
        st.write(
            "These inputs are **mean diagnostic features** (often derived from image-based nucleus measurements). "
            "If you’re not sure what a field means, hover for a quick explanation."
        )

        # Demo mode helps GitHub visitors and recruiters try the dashboard instantly
        demo_cols = st.columns([0.55, 0.45])
        with demo_cols[0]:
            use_demo = st.toggle("Use sample demo values", value=False)
        with demo_cols[1]:
            st.caption("Nice for exploring the dashboard without hunting down data.")

        default_values = [0.0] * len(FEATURES)
        if use_demo:
            # Demo only — not clinical.
            default_values = [14.0, 19.0, 92.0, 600.0, 0.10, 0.12, 0.10, 0.06, 0.18, 0.06]
            st.info("Sample values loaded (for demo purposes only).")

        with st.expander("Input fields (mean features)", expanded=True):
            cols = st.columns(2)
            inputs = []

            for i, ((label, help_text), default) in enumerate(zip(FEATURES, default_values)):
                with cols[i % 2]:
                    val = st.number_input(
                        label=label,
                        min_value=0.0,
                        value=float(default),
                        step=0.001,
                        format="%.4f",
                        help=help_text,
                    )
                    inputs.append(val)

        st.divider()

        # Run validation BEFORE letting the user predict
        checks = validate_inputs(inputs)

        if checks["fatal"]:
            st.error("Before running the model, a quick issue to fix:")
            for msg in checks["fatal"]:
                st.write(f"- {msg}")
            st.stop()

        if checks["warnings"]:
            st.warning("A few values look a bit off. This doesn’t mean they’re wrong—just double-checking:")
            for msg in checks["warnings"]:
                st.write(f"- {msg}")

            for msg in checks["notes"]:
                st.caption(msg)

            run_anyway = st.checkbox("I checked my inputs—run the assessment anyway", value=False)
        else:
            run_anyway = True  # no warnings, so allow running normally

        st.divider()

        run = st.button(
            "Generate model assessment",
            type="primary",
            disabled=not run_anyway,
            help="Runs the model using the values above. Output is an estimate, not a diagnosis.",
        )

    with right:
        st.subheader("2) Results")
        st.write("Your results will show up here with an explanation and (when available) confidence.")

        if run:
            x = np.array(inputs, dtype=float).reshape(1, -1)

            try:
                pred_class, pred_label, prob_for_pred, _ = safe_predict(model, x)
            except Exception as e:
                st.error(
                    "Something went wrong while generating the assessment. "
                    "This usually happens when the model expects a different feature format."
                )
                st.exception(e)
                st.stop()

        
            if pred_label == "Malignant":
                st.warning("**Model assessment:** patterns closer to malignant examples")
                explanation = (
                    "Based on the training data, your inputs look more similar to cases labeled malignant. "
                    "That’s not a diagnosis—it’s a statistical match to past examples."
                )
            elif pred_label == "Benign":
                st.success("**Model assessment:** patterns closer to benign examples")
                explanation = (
                    "Based on the training data, your inputs look more similar to cases labeled benign. "
                    "That’s not a diagnosis—it’s a statistical match to past examples."
                )
            else:
                st.info(f"**Model assessment:** {pred_label}")
                explanation = "The model returned a class label. Interpret it cautiously and in context."

            st.write(explanation)

            # Confidence display (if available)
            if prob_for_pred is not None:
                pct = float(prob_for_pred) * 100.0
                st.metric("Model confidence (for this output)", f"{pct:.2f}%")
                st.progress(min(max(prob_for_pred, 0.0), 1.0))
                st.caption(
                    "This confidence is the model’s internal probability estimate for the predicted class. "
                    "It’s not the same thing as clinical certainty."
                )
            else:
                st.caption("This model doesn’t provide probabilities (no predict_proba available).")

            st.divider()
            st.subheader("What you’d do next (in a real workflow)")
            st.markdown(
                """
- A clinician would interpret this alongside imaging, pathology, labs, and patient history.  
- If you’re using this for a class/teaching demo, try slightly adjusting one feature at a time to see how the model reacts.  
- If this is about real health, please talk to a licensed professional.
                """
            )


# Page: How to Use
elif page == "How to Use":
    st.subheader("How to use this prototype")
    st.markdown(
        """
This dashboard expects **10 mean diagnostic feature values**.

### Practical tips
- Use real measured values from your dataset/workflow.
- If you don’t know the values, use the **sample demo values** to explore the UI.
- The output is a **probabilistic estimate**, not a diagnosis.

### Why the dashboard warns about unusual inputs
The dashboard uses “soft” ranges to catch obvious typos (like mixing up units or pasting the wrong column).
They’re just guardrails—not medical thresholds.
        """
    )

    with st.expander("What do these features represent?"):
        st.markdown(
            """
These kinds of features often come from nucleus measurements (size, shape, texture) derived from medical imaging workflows.
They are common in classic ML teaching datasets and help demonstrate classification pipelines responsibly.
            """
        )


# Page: Model & Performance
elif page == "Model & Performance":
    st.subheader("Model & performance transparency")
    st.write(
        "If you’re going to use ML in a health-related context, showing performance and failure modes is part of responsible design."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Artifacts loaded", "Yes")
    with c2:
        st.metric("Reported accuracy", metrics.get("Accuracy", "Not available"))
    with c3:
        st.metric("Probabilities supported", "Yes" if hasattr(model, "predict_proba") else "No")

    st.divider()

    st.subheader("Confusion matrix")
    if conf_matrix is not None:
        st.pyplot(draw_confusion_matrix(conf_matrix))

        try:
            tn, fp = conf_matrix[0]
            fn, tp = conf_matrix[1]
            st.markdown(
                f"""
**Brief breakdown (from the stored test set):**
- Benign correctly identified (TN): **{tn}**
- Benign flagged as malignant (FP): **{fp}**
- Malignant missed (FN): **{fn}**
- Malignant correctly identified (TP): **{tp}**
                """
            )
        except Exception:
            st.caption("Confusion matrix loaded, but I couldn’t parse the labels cleanly.")
    else:
        st.warning("No confusion matrix file was found.")


# Page: Limitations & Ethics
else:
    st.subheader("Limitations & ethics")
    st.markdown(
        """
### Data limitations
- Trained on a specific dataset, so performance may not generalize to other populations or clinical settings.
- Uses numeric features (not raw imaging), which simplifies real-world complexity.

### Model limitations
- The model learns patterns from past labeled examples. It can be confidently wrong.
- Probability scores are not clinical truth—they’re the model’s internal estimate.

### Clinical limitations
- Does not include patient history, imaging review, pathology interpretation, or clinician judgment.
- Not validated for clinical deployment.

### Human-centered risk
- Users may over-trust an “AI answer” if the interface looks too authoritative.
- A scary label can cause anxiety, so this prototype intentionally uses calm, decision-support language.
        """
    )
