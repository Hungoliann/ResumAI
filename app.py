import os
import markdown2
import streamlit as st
from utils.resume_parser import ResumeParser
from utils.score_resume import ScoreResume
from utils.gemini_helper import gemini_resume_helper
from utils.resume_ranker_nn import ResumeRanker

# ---------------------- Streamlit UI ----------------------
st.markdown(
    """
    <div class="tight-container" style="text-align: center;">
        <h2 style='font-size: 70px; line-height: 0'>ResumAI</h2>
        <h3 style='color: #CABEFF; font-size: 30px;'><I>AI RESUME READER AND RECOMMENDER</I></h3>
        <p>This tool will match your uploaded resume with your job description and recommend ways to improve it.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------- Session State ----------------------
if "show_feedback_dialog" not in st.session_state:
    st.session_state["show_feedback_dialog"] = False

if "ai_response" not in st.session_state:
    st.session_state["ai_response"] = ""

if not os.path.exists("assets"):
    os.makedirs("assets")

# ---------------------- Load NN once (cached) ----------------------
@st.cache_resource
def load_ranker() -> ResumeRanker:
    """Load the fine-tuned ranker once and reuse across sessions."""
    return ResumeRanker('models/resume_ranker')

ranker = load_ranker()

# ---------------------- Dialog Definitions ----------------------
@st.dialog("AI Resume Feedback", width="large")
def show_ai_feedback():
    st.markdown(st.session_state["ai_response"], unsafe_allow_html=True)

@st.dialog("Score", width="large")
def show_score_and_missing_words(result, nn_score: float | None = None):
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Keyword Match Score",
            value=f"{result['score']:.1f}%",
            help="Cosine similarity between resume and job description embeddings."
        )

    with col2:
        if nn_score is not None:
            st.metric(
                label="Neural Relevance Score",
                value=f"{nn_score:.1f}%",
                delta=f"{nn_score - result['score']:+.1f}% vs keyword score",
                help="Score from a fine-tuned neural model that learned job-resume "
                     "relevance patterns beyond keyword overlap."
            )
        else:
            st.metric(label="Neural Relevance Score", value="—")

    st.divider()
    st.markdown("**Missing keywords:**")
    st.write(", ".join(sorted(result["missing"])))

# ---------------------- Helper Functions ----------------------
def parse_and_score(job_description: str, uploaded_file) -> dict | None:
    """Run the full resume parsing + cosine scoring pipeline."""
    with st.spinner("Running your resume through our system..."):
        scorer = ScoreResume(uploaded_file, job_description)
        scorer.validate().save().get_parse().score_resume()
        result = scorer.get_result()
        if "error" in result:
            st.error(result["error"])
            return None
        st.session_state["resume_text"] = result["resume_text"]
        st.session_state["job_text"] = result["job_text"]
    return result

def check_inputs(job_description: str, uploaded_file) -> bool:
    """Validate that both inputs are present before running any feature."""
    if not job_description or not uploaded_file:
        st.session_state["has_file"] = False
        return False
    st.session_state["has_file"] = True
    return True

# ---------------------- Inputs ----------------------
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt", "docx"])
job_description = st.text_area("Paste your job description here", max_chars=999999)

# ---------------------- Buttons ----------------------
col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 3, 1, 3, 1, 3, 2])

with col2:
    if st.button("Get resume match"):
        if check_inputs(job_description, uploaded_file):
            result = parse_and_score(job_description, uploaded_file)
            if result:
                show_score_and_missing_words(result, nn_score=None)

with col4:
    if st.button("Get feedback"):
        if check_inputs(job_description, uploaded_file):
            # Ensure resume is parsed first if not already in session
            if not st.session_state.get("resume_text"):
                parse_and_score(job_description, uploaded_file)
            with st.spinner("Generating AI feedback..."):
                ai_md = gemini_resume_helper(
                    st.session_state["resume_text"],
                    st.session_state["job_text"]
                )
                st.session_state["ai_response"] = markdown2.markdown(ai_md)
                st.session_state["show_feedback_dialog"] = True

with col6:
    if st.button("Neural Score", help="Score using fine-tuned neural relevance model"):
        if check_inputs(job_description, uploaded_file):
            result = parse_and_score(job_description, uploaded_file)
            if result:
                with st.spinner("Running neural ranker..."):
                    nn_score = ranker.score(
                        st.session_state["resume_text"],
                        st.session_state["job_text"]
                    )
                show_score_and_missing_words(result, nn_score=nn_score)

if st.session_state.get("has_file") is False:
    st.error("Please upload your resume and fill in the job description!")

# ---------------------- Show Feedback Dialog ----------------------
if st.session_state.get("show_feedback_dialog", False):
    show_ai_feedback()
    st.session_state["show_feedback_dialog"] = False