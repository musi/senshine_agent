
import streamlit as st
import pandas as pd
import json
import re
import numpy as np
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# Load data
@st.cache_data
def load_data():
    df_q = pd.read_excel("AdvisoryData.xlsx", sheet_name="Q")
    df_a = pd.read_excel("AdvisoryData.xlsx", sheet_name="A")
    with open("rules_to_recommendation_enriched.json", "r") as f:
        rules = json.load(f)
    return df_q, df_a, rules

df_q, df_a, rules = load_data()

# Build mappings
qid_to_question = {str(row["QID"]): row["Question"] for _, row in df_q.iterrows()}
answer_lookup = df_a.set_index(["Set_Id", "Id"])["Answer"].to_dict()

qid_to_answer_set = {str(row["QID"]): str(row["Answer Set"]) for _, row in df_q.iterrows()}
qid_to_valid_answers = {}
for qid, set_id in qid_to_answer_set.items():
    answers = [ans for (s, aid), ans in answer_lookup.items() if str(s) == set_id]
    qid_to_valid_answers[qid] = sorted(answers)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
answer_lookup_embed = {}
for qid, answers in qid_to_valid_answers.items():
    for ans in answers:
        answer_lookup_embed[ans] = embed_model.encode(ans, normalize_embeddings=True)

# Normalize text
def normalize(text):
    txt = re.sub(r"[^\w\s]", " ", text or "")
    return re.sub(r"\s+", " ", txt).strip().lower()

def best_semantic_match(user_ans, valid_answers, threshold=0.3, fuzzy_thresh=0.75):
    if not user_ans or not valid_answers:
        return None
    u_norm = normalize(user_ans)
    for ans in valid_answers:
        if normalize(ans) == u_norm:
            return ans
    for ans in valid_answers:
        if normalize(ans) in u_norm or u_norm in normalize(ans):
            return ans
    best_score = 0
    best_ans = None
    for ans in valid_answers:
        score = SequenceMatcher(None, u_norm, normalize(ans)).ratio()
        if score > best_score and score >= fuzzy_thresh:
            best_score = score
            best_ans = ans
    if best_ans:
        return best_ans
    u_vec = embed_model.encode([u_norm], normalize_embeddings=True)
    v_vecs = [answer_lookup_embed[a] for a in valid_answers if a in answer_lookup_embed]
    if not v_vecs:
        return None
    sims = cosine_similarity(u_vec, v_vecs)[0]
    best_i = int(np.argmax(sims))
    if sims[best_i] >= threshold:
        return valid_answers[best_i]
    return None

# GPT answer extraction
def extract_semantic_answers(api_key, user_input, expected_qids=None):
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-01-01-preview",
        azure_endpoint="https://chris-m9wrk3yu-eastus2.cognitiveservices.azure.com/"
    )
    qlines = []
    for qid in expected_qids or qid_to_question.keys():
        qtext = qid_to_question.get(qid, "")
        valid = qid_to_valid_answers.get(qid, [])
        example_str = f" (valid: {', '.join(valid[:3])})" if valid else ""
        qlines.append(f"{qid}: {qtext}{example_str}")
    questions_block = "\n".join(qlines)
    prompt = (
        "The user responded: \"" + user_input + "\"\n\n"
        "Here are the questions:\n" + questions_block + "\n\n"
        "Please extract answers in JSON like {'Q1': 'yes', 'Q2': 'maths'}."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4-1",
            messages=[
                {"role": "system", "content": "Map user input to known questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        parsed = json.loads(response.choices[0].message.content.strip())
        cleaned_result = {}
        for qid, answer in parsed.items():
            valid = qid_to_valid_answers.get(qid, [])
            m = best_semantic_match(str(answer), valid)
            if m:
                cleaned_result[qid] = m
        return cleaned_result
    except Exception as e:
        print("⚠️ GPT failed:", e)
        return {}

# Rule matching
def score_partial_matches(state, rules):
    results = []
    for rule in rules:
        rule_path = rule["conditions"]
        matched = [qid for qid, val in rule_path.items() if state.get(qid) == val]
        confidence = round(100 * len(matched) / len(rule_path), 1) if rule_path else 0
        results.append({
            "rcode": rule["rcode"],
            "confidence": confidence,
            "matched_qs": matched,
            "missing_qs": [q for q in rule_path if q not in matched],
            "description": rule.get("recommendation"),
            "rationale": rule.get("rationale")
        })
    return sorted(results, key=lambda x: -x["confidence"])

# Streamlit app UI
st.title("🎓 Senshine Interactive Advisor")

api_key = st.text_input("🔐 Azure OpenAI API Key", type="password")
scenario = st.text_area("✏️ Describe the learner's situation (free text):", height=200)

if "user_state" not in st.session_state:
    st.session_state.user_state = {}

if "final_r" not in st.session_state:
    st.session_state.final_r = None

if st.button("Start Advisor"):
    if not api_key or not scenario:
        st.warning("Please provide input and API key.")
    else:
        answers = extract_semantic_answers(api_key, scenario)
        st.session_state.user_state.update(answers)

        results = score_partial_matches(st.session_state.user_state, rules)
        top = results[0]

        if top["confidence"] == 100.0:
            st.session_state.final_r = top
            st.success(f"🎯 Recommendation: {top['rcode']} (100%)")
            st.write(f"📖 {top['description']}")
            st.write(f"🧠 {top['rationale']}")
            st.json({k: st.session_state.user_state[k] for k in top["matched_qs"]})
        else:
            st.warning("Partial match. Top 5 recommendations:")
            for r in results[:5]:
                st.markdown(f"- **{r['rcode']}**: {r['confidence']}%")
            if top["missing_qs"]:
                st.markdown("### 🔍 Clarify the following:")
                for qid in top["missing_qs"][:2]:
                    qtext = qid_to_question.get(qid, "Unknown question")
                    followup = st.text_input(f"{qid}: {qtext}")
                    if followup:
                        valid = qid_to_valid_answers.get(qid, [])
                        m = best_semantic_match(followup, valid)
                        if m:
                            st.session_state.user_state[qid] = m
