import streamlit as st

st.title("Sales GPT App")

question = st.text_input("Ask your question:")

if st.button("Submit"):

    try:
        # 1. SQL
        st.subheader("Generated SQL")
        sql = "SELECT 'test'"
        st.code(sql, language="sql")

        # 2. Result
        st.subheader("Result")
        result = "Test result working"
        st.write(result)

        # 3. Insight
        st.subheader("Insight")
        st.write("App is working correctly 🎉")

    except Exception as e:
        st.error(f"Error happened: {e}")