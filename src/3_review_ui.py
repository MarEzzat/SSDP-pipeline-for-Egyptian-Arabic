import streamlit as st
import json
import os

st.set_page_config(page_title="SSDP Review", layout="wide")


@st.cache_data
def load_samples():
    with open("data/synthesis_results.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_reviews():
    try:
        with open("data/reviews.json", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_reviews(reviews):
    with open("data/reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)


samples = load_samples()

# Session state
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "reviews" not in st.session_state:
    st.session_state.reviews = load_reviews()

# Skip already-reviewed samples
while (
    st.session_state.current_idx < len(samples)
    and samples[st.session_state.current_idx]["id"] in st.session_state.reviews
):
    st.session_state.current_idx += 1

# Header
st.title("🎙️ SSDP Sample Review")
reviewed_count = len(st.session_state.reviews)
st.progress(reviewed_count / len(samples))
st.write(f"Reviewed: {reviewed_count}/{len(samples)}")

if st.session_state.current_idx >= len(samples):
    st.success(" All samples reviewed!")
    st.write("Run `python src/4_export_dataset.py` to export the dataset.")
else:
    sample = samples[st.session_state.current_idx]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Sample {st.session_state.current_idx + 1}/{len(samples)}")
        st.write(f"**Text**: {sample['text']}")
        st.write(f"**Duration**: {sample['duration']:.2f}s")
        st.write(f"**Domain**: {sample.get('domain', 'N/A')}")

        if os.path.exists(sample["audio_path"]):
            with open(sample["audio_path"], "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/wav")
        else:
            st.error(f"Audio file not found: {sample['audio_path']}")

        if sample.get("flags"):
            st.warning(f" Auto-flagged: {', '.join(sample['flags'])}")

    with col2:
        st.subheader("Review Decision")
        decision = st.radio(
            "Quality",
            [" Approve", " Reject", " Flag for manual check"],
            key=f"decision_{st.session_state.current_idx}"
        )
        notes = st.text_area("Notes (optional)", key=f"notes_{st.session_state.current_idx}")

        col_submit, col_skip = st.columns(2)
        with col_submit:
            if st.button("Submit & Next", type="primary"):
                st.session_state.reviews[sample["id"]] = {
                    "decision": decision.replace("*** ", "").replace("*** ", "").replace("*** ", ""),
                    "notes": notes
                }
                save_reviews(st.session_state.reviews)
                st.session_state.current_idx += 1
                st.rerun()

        with col_skip:
            if st.button("Skip"):
                st.session_state.current_idx += 1
                st.rerun()

# Sidebar stats
with st.sidebar:
    st.subheader(" Review Stats")
    reviews = st.session_state.reviews
    approved = sum(1 for v in reviews.values() if "Approve" in v["decision"])
    rejected = sum(1 for v in reviews.values() if "Reject" in v["decision"])
    flagged  = sum(1 for v in reviews.values() if "Flag" in v["decision"])
    st.metric(" Approved", approved)
    st.metric(" Rejected", rejected)
    st.metric(" Flagged", flagged)
    st.metric(" Remaining", len(samples) - reviewed_count)