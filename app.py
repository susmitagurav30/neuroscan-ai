
from pathlib import Path
from datetime import datetime
import json
import tempfile
import sqlite3
import hashlib
import secrets
import hmac

import streamlit as st
import pandas as pd
from PIL import Image
from src.predict import predict_image, CLASS_NAMES

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NeuroScan AI | Brain MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_image" not in st.session_state:
    st.session_state.last_image = None

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #080d19;
    --panel: #101a2d;
    --panel2: #14213a;
    --border: #263753;
    --cyan: #54e4e2;
    --blue: #7b9cff;
    --muted: #9babc4;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #122343 0%, transparent 38%),
        var(--bg);
    color: #f2f6ff;
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f2f6ff !important;
}

[data-testid="stSidebar"] {
    background: #0b1323;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: #e5edff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

.hero {
    background: linear-gradient(115deg, #142c49, #122039 60%, #132c3d);
    border: 1px solid #29435d;
    border-radius: 22px;
    padding: 28px 30px;
    margin-bottom: 24px;
}

.eyebrow {
    color: var(--cyan);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(26px, 4vw, 38px);
    font-weight: 700;
    margin: 8px 0;
    color: white;
}

.hero-subtitle {
    color: #b5c5dc;
    font-size: 15px;
    line-height: 1.7;
}

.panel {
    background: linear-gradient(145deg, #111d32, #0d1728);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 19px;
    font-weight: 600;
    margin-bottom: 5px;
}

.muted {
    color: var(--muted);
    font-size: 13px;
}

.metric-card {
    background: linear-gradient(145deg, #14243c, #101a2b);
    border: 1px solid #2b405e;
    border-radius: 16px;
    padding: 18px;
    min-height: 118px;
}

.metric-label {
    color: #a7b8d0;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #f5f8ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px;
    font-weight: 700;
    margin-top: 8px;
}

.prediction-result {
    background: linear-gradient(120deg, #123d40, #112c39);
    border: 1px solid #286365;
    border-radius: 16px;
    padding: 20px;
    margin: 12px 0;
}

.prediction-name {
    color: #8af4e8;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 25px;
    font-weight: 700;
}

div.stButton > button {
    background: linear-gradient(100deg, #53e0dc, #75caff);
    color: #07131f;
    border: 0;
    border-radius: 11px;
    min-height: 46px;
    font-weight: 700;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    border: 1px solid #a3fff9;
    color: #07131f;
    filter: brightness(1.08);
}

[data-testid="stFileUploader"] {
    background: #111e32;
    border: 1px dashed #405a7a;
    border-radius: 13px;
    padding: 12px;
}

[data-testid="stMetric"] {
    background: #111e32;
    border: 1px solid #273b58;
    padding: 15px;
    border-radius: 13px;
}

hr {
    border-color: #263753;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

/* Readable upload area */
[data-testid="stFileUploaderDropzone"] {
    background: #14213a !important;
    border: 1px dashed #54e4e2 !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #f2f6ff !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #54e4e2 !important;
    color: #08111e !important;
    border: none !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: #cbd8ed !important;
}

/* Bright, readable metric cards */
[data-testid="stMetric"] {
    background: #14213a !important;
    border: 1px solid #304766 !important;
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] label *,
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #f2f6ff !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] button {
    background: #14213a !important;
    color: #54e4e2 !important;
    border: 1px solid #54e4e2 !important;
    border-radius: 11px !important;
    font-weight: 600 !important;
}

[data-testid="stDownloadButton"] button p {
    color: #54e4e2 !important;
}

[data-testid="stDownloadButton"] button:hover {
    background: #203451 !important;
}

/* Improve information and status text */
[data-testid="stAlert"] {
    background: #14213a !important;
    color: #f2f6ff !important;
    border: 1px solid #304766 !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: #f2f6ff !important;
}

/* Ensure chart and caption text remains readable */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #b8c9e1 !important;
}

/* Remove the white top header and toolbar background */
header[data-testid="stHeader"] {
    background: #080d19 !important;
}

[data-testid="stToolbar"] {
    background: transparent !important;
}

[data-testid="stDecoration"] {
    background: transparent !important;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #080d19 !important;
}

/* Make the uploaded filename card dark */

/* Uploaded filename: dark background and visible text */

[data-testid="stFileUploaderFile"] {
    background-color: #14213a !important;
    border: 1px solid #54e4e2 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}



[data-testid="stFileUploaderFile"] svg {
    fill: #54e4e2 !important;
    color: #54e4e2 !important;
}

/* Upload box text and helper details */
[data-testid="stFileUploader"] section {
    background: #101a2d !important;
    border-color: #54e4e2 !important;
}

[data-testid="stFileUploader"] small {
    color: #cbd8ed !important;
}

/* Strong override for the uploaded filename row in Streamlit 1.66.1 */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div {
    background-color: #14213a !important;
    color: #ffffff !important;
    border: 1px solid #54e4e2 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] p,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg {
    color: #54e4e2 !important;
    fill: #54e4e2 !important;
}

/* Make chart area match the dashboard */
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"],
[data-testid="stPyplotChart"] {
    background: #101a2d !important;
    border: 1px solid #263753 !important;
    border-radius: 14px !important;
    padding: 10px !important;
}
/* NeuroScan AI: Dark theme  */
.stApp {
    background-color: #080d19;
    color: #f1f5f9;
}

.stTextInput input,
.stTextInput input:focus,
.stTextInput input:hover,
.stPasswordInput input {
    background-color: #111c30 !important;
    color: #ffffff !important;
    border: 1px solid #34445f !important;
    border-radius: 10px !important;
    caret-color: #22d3ee !important;
}

.stTextInput label,
.stTextInput label p,
.stTabs [data-baseweb="tab"] p {
    color: #e2e8f0 !important;
}

.stTextInput input::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #ef4444, #06b6d4) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stButton > button p {
    color: #ffffff !important;
}

.stButton > button:hover {
    filter: brightness(1.12);
    border: 1px solid #67e8f9 !important;
}

/* NeuroScan navy-blue submit buttons */
.stApp [data-testid="stFormSubmitButton"] button,
.stApp [data-testid="stFormSubmitButton"] button:hover,
.stApp [data-testid="stFormSubmitButton"] button:focus {
    background: #142b49 !important;
    background-color: #142b49 !important;
    color: #ffffff !important;
    border: 1px solid #2b527b !important;
    border-radius: 10px !important;
    opacity: 1 !important;
    box-shadow: none !important;
}

/* Keep button text clearly visible */
.stApp [data-testid="stFormSubmitButton"] button *,
.stApp [data-testid="stFormSubmitButton"] button p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}

/* Navy-blue hover effect */
.stApp [data-testid="stFormSubmitButton"] button:hover {
    background: #1b3b60 !important;
    border-color: #22d3ee !important;
}

/* NeuroScan navy theme: MRI uploader */
.stApp [data-testid="stFileUploader"] section {
    background-color: #101d31 !important;
    border: 1px dashed #22d3ee !important;
    border-radius: 12px !important;
}

/* Uploaded file row */
.stApp [data-testid="stFileUploaderFile"],
.stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background-color: #142b49 !important;
    border: 1px solid #2b527b !important;
    border-radius: 10px !important;
}

/* File name and details */
.stApp [data-testid="stFileUploaderFile"] *,
.stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
    color: #f1f5f9 !important;
}

/* Upload icon and remove-file button */
.stApp [data-testid="stFileUploader"] button {
    background-color: #142b49 !important;
    color: #ffffff !important;
    border: 1px solid #2b527b !important;
}

/* Hover effect */
.stApp [data-testid="stFileUploader"] button:hover {
    background-color: #1b3b60 !important;
    border-color: #22d3ee !important;
}


/* Dark theme for data tables */
[data-testid="stDataFrame"],
[data-testid="stDataEditor"],
[data-testid="stTable"] {
    background: #101a2d !important;
    color: #f2f6ff !important;
    border: 1px solid #263753 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Dark table cells and readable text */
[data-testid="stDataFrame"] *,
[data-testid="stTable"] * {
    color: #f2f6ff !important;
}

/* Dark chart containers */
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"],
[data-testid="stPyplotChart"],
[data-testid="stAltairChart"],
[data-testid="stPlotlyChart"] {
    background: #101a2d !important;
    border: 1px solid #263753 !important;
    border-radius: 14px !important;
    padding: 10px !important;
}

/* Dark backgrounds for chart SVGs and chart canvases */
[data-testid="stVegaLiteChart"] canvas,
[data-testid="stVegaLiteChart"] svg,
[data-testid="stArrowVegaLiteChart"] canvas,
[data-testid="stArrowVegaLiteChart"] svg {
    background: transparent !important;
}

/* Dark dataframe toolbar */
[data-testid="stDataFrame"] button {
    background: #142b49 !important;
    color: #f2f6ff !important;
    border-color: #304766 !important;
}

/* Make filenames and titles visible in expandable descriptions */
[data-testid="stExpander"] {
    background: #101a2d !important;
    border: 1px solid #304766 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: #f2f6ff !important;
    -webkit-text-fill-color: #f2f6ff !important;
    opacity: 1 !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #101a2d !important;
    color: #f2f6ff !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIMPLE MULTI-USER AUTHENTICATION ----------------
# Demo-ready local authentication. For a public production deployment,
# use a managed auth provider and persistent database.
AUTH_DB = Path(__file__).parent / "users.db"


def init_auth_db():
    with sqlite3.connect(AUTH_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def hash_password(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310000)
    return salt.hex(), digest.hex()


def create_user(username, password):
    salt, password_hash = hash_password(password)
    try:
        with sqlite3.connect(AUTH_DB) as conn:
            conn.execute(
                "INSERT INTO users (username, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, salt, password_hash, datetime.now().isoformat(timespec="seconds")),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username, password):
    with sqlite3.connect(AUTH_DB) as conn:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    if not row:
        return False
    _, attempted_hash = hash_password(password, row[0])
    return hmac.compare_digest(attempted_hash, row[1])


init_auth_db()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.authenticated:
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">SECURE WORKSPACE</div>
        <div class="hero-title">🧠 NeuroScan AI</div>
        <div class="hero-subtitle">Sign in or create an account to access the Brain MRI Analysis Studio.</div>
    </div>
    """, unsafe_allow_html=True)

    login_tab, signup_tab = st.tabs(["Sign in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Sign in", use_container_width=True)
        if login_submit:
            username = login_username.strip()
            if verify_user(username, login_password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.history = []
                st.session_state.last_result = None
                st.session_state.last_image = None
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", key="signup_username")
            new_password = st.text_input("Create password (at least 8 characters)", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            signup_submit = st.form_submit_button("Create account", use_container_width=True)
        if signup_submit:
            username = new_username.strip()
            if len(username) < 3:
                st.error("Username must contain at least 3 characters.")
            elif len(new_password) < 8:
                st.error("Password must contain at least 8 characters.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif create_user(username, new_password):
                st.success("Account created. You can now sign in.")
            else:
                st.error("That username already exists. Please choose another.")

    st.caption("Educational project only. Do not upload identifiable patient records or use results for medical decisions.")
    st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🧠 NeuroScan AI")
    st.caption("BRAIN MRI INTELLIGENCE")
    st.write(f"Signed in as **{st.session_state.username}**")
    if st.button("Sign out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.last_image = None
        st.rerun()
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Prediction History", "About Model"],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("### Model information")
    st.write("**Architecture:** ResNet18")
    st.write("**Classes:** 4")
    st.write("**Input:** MRI image")
    st.write("**Image size:** 224 × 224")

    st.divider()
    st.caption("Educational project • Not for clinical diagnosis")

# ---------------- HELPERS ----------------
def metric_card(label, value, detail):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="muted">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def reset_current_result():
    st.session_state.last_result = None
    st.session_state.last_image = None


# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">AI-POWERED IMAGE CLASSIFICATION</div>
        <div class="hero-title">Brain MRI Analysis Studio</div>
        <div class="hero-subtitle">
            Upload an MRI image, run the trained model, and explore
            its predicted class and class scores in one workspace.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Model", "ResNet18", "Fine-tuned CNN")

    with c2:
        metric_card("Categories", "04", "Classification classes")

    with c3:
        metric_card("Test accuracy", "93.75%", "On your current 1,600-image test set")

    with c4:
        metric_card("Images analyzed", str(len(st.session_state.history)),
                    "This session only")

    st.write("")

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">01 · MRI Image</div>
            <div class="muted">Choose an image to preview before analysis.</div>
        </div>
        """, unsafe_allow_html=True)

        
        uploaded_files = st.file_uploader(
            "Upload MRI images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Upload one or more MRI images.",
            key="mri_upload",
        )

        if uploaded_files:
            st.caption(f"{len(uploaded_files)} image(s) selected")

            # Preview all selected images
            preview_cols = st.columns(min(len(uploaded_files), 3))

            for index, uploaded in enumerate(uploaded_files):
                try:
                    image = Image.open(uploaded).convert("RGB")
                    with preview_cols[index % len(preview_cols)]:
                        st.image(
                            image,
                            caption=uploaded.name,
                            use_container_width=True,
                        )
                except Exception as error:
                    st.error(f"Could not open {uploaded.name}: {error}")

            if st.button(
                "✦ Analyze All MRI Images",
                type="primary",
                use_container_width=True,
            ):
                batch_results = []
                progress = st.progress(0)

                for index, uploaded in enumerate(uploaded_files):
                    temp_path = None

                    try:
                        image = Image.open(uploaded).convert("RGB")

                        with tempfile.NamedTemporaryFile(
                            suffix=".jpg", delete=False
                        ) as temp:
                            temp_path = Path(temp.name)

                        image.save(temp_path, format="JPEG")

                        with st.spinner(
                            f"Analyzing {index + 1}/{len(uploaded_files)}: "
                            f"{uploaded.name}"
                        ):
                            result = predict_image(temp_path)

                        item = {
                            "image": uploaded.name,
                            "predicted_class": result["predicted_class"],
                            "confidence": result["confidence"],
                            "class_probabilities": result[
                                "class_probabilities"
                            ],
                        }
                        batch_results.append(item)

                        st.session_state.history.insert(0, {
                            "time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            **item,
                        })

                    except Exception as error:
                        st.error(
                            f"Prediction failed for {uploaded.name}: {error}"
                        )

                    finally:
                        if temp_path is not None:
                            try:
                                temp_path.unlink(missing_ok=True)
                            except OSError:
                                pass

                    progress.progress((index + 1) / len(uploaded_files))

                st.session_state.batch_results = batch_results

                if batch_results:
                    latest = batch_results[-1]
                    st.session_state.last_result = {
                        "predicted_class": latest["predicted_class"],
                        "confidence": latest["confidence"],
                        "class_probabilities": latest["class_probabilities"],
                    }
                    st.session_state.last_image = latest["image"]

                    st.success(
                        f"Completed analysis for {len(batch_results)} image(s)."
                    )
                    st.rerun()

        else:
            st.info("Upload one or more MRI images to begin.")


    with right:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">02 · Analysis Results</div>
            <div class="muted">Your latest model prediction appears here.</div>
        </div>
        """, unsafe_allow_html=True)

        result = st.session_state.last_result
        
        batch_results = st.session_state.get("batch_results", [])

        if batch_results:
            st.markdown("#### Results for all uploaded images")

            results_df = pd.DataFrame([
                {
                    "Image": item["image"],
                    "Predicted class": item["predicted_class"],
                    "Model score (%)": round(
                        float(item["confidence"]), 2
                    ),
                }
                for item in batch_results
            ])

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download results (CSV)",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="neuroscan_results.csv",
                mime="text/csv",
                use_container_width=True,
            )


        if result is None:
            st.markdown("""
            <div class="panel" style="text-align:center; padding:48px 20px;">
                <div style="font-size:48px;">⌁</div>
                <div class="panel-title">Awaiting MRI image</div>
                <div class="muted">
                    Upload an image and select Analyze MRI to view results.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            
            batch_results = st.session_state.get("batch_results", [])

            if not batch_results:
                batch_results = [{
                    "image": st.session_state.last_image or "MRI image",
                    "predicted_class": result["predicted_class"],
                    "confidence": result["confidence"],
                    "class_probabilities": result["class_probabilities"],
                }]

            st.markdown("### Predictions & Class Scores")

            for item in batch_results:
                predicted = item["predicted_class"]
                score = float(item["confidence"])
                probabilities = item["class_probabilities"]

                st.markdown(
                    f"""
                    <div style="
                        background:linear-gradient(120deg,#123d40,#112c39);
                        border:1px solid #286365;
                        border-radius:14px;
                        padding:16px;
                        margin:14px 0 8px 0;">
                        <div style="color:#8bdde5;font-size:12px;">
                            IMAGE: {item['image']}
                        </div>
                        <div style="
                            color:#8af4e8;
                            font-size:23px;
                            font-weight:700;
                            margin-top:6px;">
                            {predicted.replace('_', ' ').title()}
                        </div>
                        <div style="color:#c5d7ed;margin-top:5px;">
                            Model score: {score:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Class score distribution**")

                for class_name, class_score in probabilities.items():
                    value = max(0.0, min(100.0, float(class_score)))
                    label = class_name.replace("_", " ").title()

                    st.markdown(
                        f"""
                        <div style="margin:10px 0 14px 0;">
                            <div style="
                                display:flex;
                                justify-content:space-between;
                                color:#e6efff;
                                font-size:13px;
                                margin-bottom:5px;">
                                <span>{label}</span>
                                <span>{value:.2f}%</span>
                            </div>
                            <div style="
                                background:#263753;
                                height:12px;
                                border-radius:8px;
                                overflow:hidden;">
                                <div style="
                                    background:linear-gradient(
                                        90deg,#22d3ee,#54e4e2
                                    );
                                    width:{value}%;
                                    height:12px;
                                    border-radius:8px;">
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Keep the existing educational information and report code working.
            
            predicted_class = result["predicted_class"]
            confidence = float(result["confidence"])



            # General educational information, not an individual diagnosis.
            tumor_information = {
                "glioma": {
                    "title": "Glioma",
                    "description": "A group of tumors that arise from glial cells in the brain or spinal cord. The behavior and treatment depend on the specific type and grade.",
                    "symptoms": "Possible symptoms include persistent or worsening headache, seizures, nausea or vomiting, weakness, changes in speech or vision, and balance or memory difficulties.",
                },
                "meningioma": {
                    "title": "Meningioma",
                    "description": "A tumor that develops from the membranes surrounding the brain and spinal cord. Many grow slowly, and some cause no symptoms; behavior varies by type and location.",
                    "symptoms": "Possible symptoms include headache, seizures, changes in vision or hearing, weakness or numbness, and problems with balance. Symptoms depend on the tumor's location and size.",
                },
                "pituitary": {
                    "title": "Pituitary-region tumor category",
                    "description": "This model label represents images in the dataset's pituitary category. It cannot confirm the exact tumor type, hormone activity, or clinical diagnosis.",
                    "symptoms": "Depending on the cause, possible symptoms may include headaches, changes in peripheral vision, menstrual or fertility changes, unusual milk production, or changes in growth, energy, or other hormone-related functions.",
                },
                "notumor": {
                    "title": "No-tumor dataset category",
                    "description": "The model classified this image as the dataset's no-tumor class. This does not prove that the scan is normal or rule out disease.",
                    "symptoms": "No tumor-related symptoms can be inferred from this model label. If symptoms or concerns exist, a qualified clinician should review the scan and clinical history.",
                },
            }
            info = tumor_information.get(predicted_class.lower(), {
                "title": predicted_class.replace("_", " ").title(),
                "description": "No additional educational information is configured for this class.",
                "symptoms": "Symptoms cannot be determined from this model output.",
            })

            
            st.markdown("### About predicted category")

            batch_results = st.session_state.get("batch_results", [])

            if batch_results:
                for item in batch_results:
                    predicted_class_item = item["predicted_class"]

                    info_item = tumor_information.get(
                        predicted_class_item.lower(),
                        {
                            "title": predicted_class_item.replace(
                                "_", " "
                            ).title(),
                            "description": (
                                "No additional educational information "
                                "is configured for this class."
                            ),
                            "symptoms": (
                                "Symptoms cannot be determined from "
                                "this model output."
                            ),
                        },
                    )

                    with st.expander(
                        
                        f"{item['image']} — {info_item['title']}",
                        expanded=True,
                    ):
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #162640;
                                color: #ffffff;
                                padding: 12px;
                                border-radius: 8px;
                                border: 1px solid #26D3BE;
                                margin-bottom: 12px;
                            ">
                                <strong style="color: #26D3BE;">Image filename</strong><br>
                                <span style="color: #FFFFFF; font-size: 16px;">
                                    {item['image']}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.write("**Predicted category:**",
                                 info_item["title"])
                        st.write("**Model score:**",
                                 f"{float(item['confidence']):.2f}%")
                        st.write("**Description:**",
                                 info_item["description"])
                        st.write("**Possible symptoms (not guaranteed):**")
                        st.write(info_item["symptoms"])
            else:
                st.info("Analyze images to see their category descriptions.")

            st.info(
                "These descriptions are general educational information, "
                "not a diagnosis. Model predictions cannot confirm or rule "
                "out a brain tumor. Consult a qualified medical professional."
            )

        

            
            # Prepare a report for every analyzed image
            report_items = st.session_state.get("batch_results", [])

            if not report_items and result is not None:
                report_items = [{
                    "image": st.session_state.last_image or "MRI image",
                    "predicted_class": result["predicted_class"],
                    "confidence": result["confidence"],
                    "class_probabilities": result["class_probabilities"],
                }]

            reports = [
                {
                    "image": item["image"],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "predicted_class": item["predicted_class"],
                    "model_confidence_percent": float(item["confidence"]),
                    "class_scores_percent": item["class_probabilities"],
                    "notice": (
                        "Educational classification only; "
                        "not a medical diagnosis."
                    ),
                }
                for item in report_items
            ]

            # Download all image reports as JSON
            st.download_button(
                "⬇ Download All Reports (JSON)",
                data=json.dumps(reports, indent=2),
                file_name="NeuroScan_All_MRI_Reports.json",
                mime="application/json",
                use_container_width=True,
            )

            # Build a text report for every image
            report_text = [
                "NEUROSCAN AI",
                "BRAIN MRI IMAGE CLASSIFICATION REPORT",
                "=" * 45,
                f"Report date: {datetime.now().strftime('%d-%m-%Y')}",
                f"Total images analyzed: {len(report_items)}",
                "",
            ]

            for index, item in enumerate(report_items, start=1):
                report_text.extend([
                    "-" * 45,
                    f"IMAGE {index}: {item['image']}",
                    "-" * 45,
                    f"Predicted category: "
                    f"{item['predicted_class'].replace('_', ' ').title()}",
                    f"Model score: {float(item['confidence']):.2f}%",
                    "",
                    "CLASS SCORES:",
                ])

                for class_name, score in item["class_probabilities"].items():
                    report_text.append(
                        f"{class_name.replace('_', ' ').title()}: "
                        f"{float(score):.2f}%"
                    )

                report_text.append("")

            report_text.extend([
                "=" * 45,
                "DISCLAIMER",
                "This is an educational AI classification report, not a "
                "medical diagnosis. Model confidence is not a clinically "
                "validated probability. Consult a qualified medical "
                "professional for interpretation of MRI images.",
                "",
                "Generated by NeuroScan AI",
            ])

            st.download_button(
                "📄 Download All Reports (TXT)",
                data="\n".join(report_text),
                file_name="NeuroScan_MRI_Reports.txt",
                mime="text/plain",
                use_container_width=True,
            )


            
            report_text = f"""
NEUROSCAN AI
BRAIN MRI IMAGE CLASSIFICATION REPORT
======================================

Image filename: {st.session_state.last_image}
Report date: {datetime.now().strftime("%d-%m-%Y")}
Report time: {datetime.now().strftime("%I:%M:%S %p")}

MODEL INFORMATION
-----------------
Architecture: ResNet18 (fine-tuned)
Input size: 224 x 224 pixels

ANALYSIS RESULTS
----------------
Predicted category: {predicted_class.replace("_", " ").title()}
Model confidence: {confidence:.2f}%

CLASS SCORES
------------
"""

            for class_name, score in result["class_probabilities"].items():
                report_text += (
                    f"{class_name.replace('_', ' ').title()}: "
                    f"{float(score):.2f}%\n"
                )

            report_text += """
DISCLAIMER
----------
This is an educational AI classification report, not a medical
diagnosis or a radiologist's report. Model confidence is not a
clinically validated probability. Please consult a qualified
medical professional for interpretation of MRI images.

Generated by NeuroScan AI
"""


            if st.button("↻  Clear current result", use_container_width=True):
                def reset_current_result():
                    st.session_state.last_result = None
                    st.session_state.last_image = None
                    st.session_state.batch_results = []
                st.rerun()

    st.divider()

    st.markdown("### Supported categories")

    class_columns = st.columns(4)
    descriptions = {
        "glioma": "A tumor category in the training dataset.",
        "meningioma": "A tumor category in the training dataset.",
        "notumor": "The dataset's no-tumor category.",
        "pituitary": "A tumor category in the training dataset.",
    }

    for column, class_name in zip(class_columns, CLASS_NAMES):
        with column:
            st.markdown(f"""
            <div class="panel">
                <div style="color:#54e4e2;font-size:13px;">MRI CLASS</div>
                <div class="panel-title">
                    {class_name.replace("_", " ").title()}
                </div>
                <div class="muted">{descriptions.get(class_name, "Model class.")}</div>
            </div>
            """, unsafe_allow_html=True)

    st.warning(
        "Important: This educational model is not validated for clinical use. "
        "Predictions and confidence scores must not be used to diagnose, "
        "exclude, or treat a brain tumor. Consult a qualified medical professional."
    )

# ---------------- HISTORY ----------------
elif page == "Prediction History":

    st.markdown("## Prediction History")
    st.caption("Predictions made during the current Streamlit session.")

    history = st.session_state.history

    if not history:
        st.info("No predictions yet. Visit Dashboard to analyze an MRI image.")
    else:
        df = pd.DataFrame([
            {
                "Time": item["time"],
                "Image": item["image"],
                "Predicted class": item["predicted_class"],
                "Model score (%)": item["confidence"],
            }
            for item in history
        ])

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Download history as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        st.download_button(
            "⬇ Download full history as JSON",
            data=json.dumps(history, indent=2),
            file_name="prediction_history.json",
            mime="application/json",
        )

        if st.button("Clear prediction history"):
            st.session_state.history = []
            st.rerun()

# ---------------- MODEL INFORMATION ----------------
elif page == "About Model":

    st.markdown("## About the Model")
    st.markdown("""
    <div class="panel">
        <div class="panel-title">ResNet18 · Fine-tuned classifier</div>
        <p class="muted">
            This project uses a ResNet18 convolutional neural network
            fine-tuned to classify MRI images into four dataset categories.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model details")

    details = {
        "Architecture": "ResNet18",
        "Input image size": "224 × 224 pixels",
        "Number of classes": len(CLASS_NAMES),
        "Preprocessing": "RGB, resize, ImageNet normalization",
        "Inference framework": "PyTorch",
        "Interface": "Streamlit",
        "API": "FastAPI (separate backend)",
    }

    st.table(pd.DataFrame(
        list(details.items()),
        columns=["Property", "Value"],
    ))

    st.markdown("### Evaluation results")

    st.info(
        "The model achieved 93.75% accuracy on the current test set "
        "(1,500 correct out of 1,600 images). This is a dataset result, "
        "not evidence of clinical performance."
    )

    st.markdown("""
    **Limitations**
    - Dataset accuracy does not guarantee correct predictions on new hospitals,
      scanners, or patient populations.
    - Confidence scores are not necessarily calibrated probabilities.
    - The model classifies images; it does not segment or locate a tumor.
    - Independent patient-level evaluation and clinical validation are needed
      before any medical use.
    """)

    st.markdown("### Classes")
    for class_name in CLASS_NAMES:
        st.write(f"• {class_name.replace('_', ' ').title()}")

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "NEUROSCAN AI  ·  Student MLOps Project  ·  "
    "For educational and research demonstration only"
)
