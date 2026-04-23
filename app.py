import streamlit as st

st.title("Sales GPT App")

# -------- GPT (Gemini) --------
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

def nl_to_sql(question):
    prompt = f"""
Convert this question into SQL:
Question: {question}

Only return SQL.
"""
    response = model.generate_content(prompt)
    return response.text

# -------- UI --------
question = st.text_input("Ask your question:")

if st.button("Submit"):
    try:
        sql = nl_to_sql(question)

        st.subheader("Generated SQL")
        st.code(sql, language="sql")

        st.success("GPT working ✅")

    except Exception as e:
        st.error(f"Error: {e}")