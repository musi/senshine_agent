
import streamlit as st
import pandas as pd
import json
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from difflib import SequenceMatcher

# Load Excel and JSON files
@st.cache_data
def load_data():
    df_q = pd.read_excel("AdvisoryData.xlsx", sheet_name="Q")
    df_a = pd.read_excel("AdvisoryData.xlsx", sheet_name="A")
    with open("rules_to_recommendation_enriched.json", "r") as f:
        rules = json.load(f)
    return df_q, df_a, rules

st.title("🎓 Senshine Interactive Advisor")
st.write("Describe the learner's situation in natural language and get personalized recommendations.")

api_key = st.text_input("🔐 Azure OpenAI API Key", type="password")
user_input = st.text_area("✏️ Learner Scenario:", height=200)

if st.button("Get Recommendation"):
    if not api_key or not user_input:
        st.warning("Please provide both API key and scenario.")
    else:
        st.info("Processing... Please wait.")

        # Placeholder: replace with your logic
        st.success("📌 Final recommendation: R001 (100%)")
        st.write("📖 Description: Example recommendation text.")
        st.write("🧠 Rationale: Example rationale.")
        st.markdown("**Matched Questions:**")
        st.json({"Q1": "Yes", "Q2": "Needs help writing"})
