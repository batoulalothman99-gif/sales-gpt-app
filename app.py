import time
import pandas as pd
import streamlit as st
import google.generativeai as genai
from google.cloud import bigquery
from google.oauth2 import service_account
import streamlit.components.v1 as components

# =========================
# Page config
# =========================
st.set_page_config(page_title="Sales GPT App", layout="wide")
st.title("Sales GPT App")
st.write("Ask any question about the sales data warehouse.")

# =========================
# Constants
# =========================
PROJECT_ID = "project1-488217"
DATASET_ID = "dw_sales"

# =========================
# Helpers
# =========================
def clean_sql(sql: str) -> str:
    if not sql:
        return ""

    sql = sql.strip()

    if sql.startswith("```sql"):
        sql = sql.replace("```sql", "", 1).strip()
    if sql.startswith("```"):
        sql = sql.replace("```", "", 1).strip()
    if sql.endswith("```"):
        sql = sql[:-3].strip()

    return sql


@st.cache_resource
def get_gemini_model():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        st.error(f"Gemini configuration error: {e}")
        return None


@st.cache_resource
def get_bigquery_client():
    try:
        credentials = service_account.Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"])
        )
        return bigquery.Client(credentials=credentials, project=PROJECT_ID)
    except Exception as e:
        st.error(f"BigQuery client configuration error: {e}")
        return None


# =========================
# Initialize services safely
# =========================
model = get_gemini_model()
client = get_bigquery_client()

# =========================
# Schema context for GPT
# =========================
DW_SCHEMA_CONTEXT = f"""
You are a SQL expert for Google BigQuery.

Project: {PROJECT_ID}
Dataset: {DATASET_ID}

Tables:

1) `{PROJECT_ID}.{DATASET_ID}.Fact_sales_updated`
Columns:
- `Order ID`
- `Customer ID`
- `Product ID`
- `Order Date`
- `Sales`
- `TotalSales`
- `ShippingCost`
- `PaymentMethod`
- `OrderStatus`
- `LocationKey`

2) `{PROJECT_ID}.{DATASET_ID}.dim_customer`
Columns:
- `Customer ID`
- `Customer Name`
- `Segment`
- `CustomerKey`

3) `{PROJECT_ID}.{DATASET_ID}.dim_product`
Columns:
- `Product ID`
- `Product Name`
- `Category`
- `Sub-Category`
- `ProductKey`

4) `{PROJECT_ID}.{DATASET_ID}.dim_date`
Columns:
- `DateKey`
- `FullDate`
- `Year`
- `Quarter`
- `Month`
- `Day`

5) `{PROJECT_ID}.{DATASET_ID}.dim_location`
Columns:
- `LocationKey`
- `City`
- `State`
- `Postal Code`
- `Region`

Rules:
- Use BigQuery SQL only
- Always use fully qualified table names
- Use backticks for column names with spaces
- Only return valid SQL
- Do not return explanations
- Do not return markdown
"""

# =========================
# Functions
# =========================
def nl_to_sql(question: str) -> str:
    if model is None:
        raise ValueError("Gemini model is not initialized.")

    prompt = f"""
{DW_SCHEMA_CONTEXT}

Convert the following user question into a valid BigQuery SQL query.

Question: "{question}"

Return only SQL.
Do not explain anything.
"""
    response = model.generate_content(prompt)

    if not hasattr(response, "text") or not response.text:
        raise ValueError("Gemini did not return SQL text.")

    sql = clean_sql(response.text)
    return sql


def run_query(sql: str) -> pd.DataFrame:
    if client is None:
        raise ValueError("BigQuery client is not initialized.")

    query_job = client.query(sql)
    df = query_job.to_dataframe()
    return df


def build_prompt(df: pd.DataFrame, user_question: str) -> str:
    if df.empty:
        result_text = "The query returned no rows."
    else:
        # لو الداتا كبيرة، نقصها حتى ما يصير حمل كبير على Gemini
        preview_df = df.head(20)
        result_text = preview_df.to_markdown(index=False)

    prompt = f"""
You are a business analyst.

The user asked:
{user_question}

The SQL result is:
{result_text}

Please provide your answer in exactly this format:

Summary:
[2-3 sentences]

Insight:
[2-3 sentences]

Recommendation:
[2-3 sentences]
"""
    return prompt


def generate_insight(prompt: str, retry_count: int = 3) -> str:
    if model is None:
        raise ValueError("Gemini model is not initialized.")

    for attempt in range(retry_count):
        try:
            response = model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            raise ValueError("Gemini returned an empty insight response.")
        except Exception:
            if attempt < retry_count - 1:
                time.sleep(2)
            else:
                raise


def parse_insight(raw_text: str):
    summary = ""
    insight = ""
    recommendation = ""
    current_section = None

    for line in raw_text.splitlines():
        clean_line = line.strip()

        if clean_line.lower().startswith("summary:"):
            current_section = "summary"
            summary += clean_line[len("summary:"):].strip() + " "
        elif clean_line.lower().startswith("insight:"):
            current_section = "insight"
            insight += clean_line[len("insight:"):].strip() + " "
        elif clean_line.lower().startswith("recommendation:"):
            current_section = "recommendation"
            recommendation += clean_line[len("recommendation:"):].strip() + " "
        else:
            if current_section == "summary":
                summary += clean_line + " "
            elif current_section == "insight":
                insight += clean_line + " "
            elif current_section == "recommendation":
                recommendation += clean_line + " "

    return {
        "summary": summary.strip(),
        "insight": insight.strip(),
        "recommendation": recommendation.strip()
    }

# =========================
# Sidebar debug info
# =========================
with st.sidebar:
    st.subheader("System Status")
    st.write("Gemini:", "✅ Ready" if model else "❌ Not Ready")
    st.write("BigQuery:", "✅ Ready" if client else "❌ Not Ready")

# =========================
# UI
# =========================
question = st.text_input("Ask your question in English:")

if st.button("Submit"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif model is None or client is None:
        st.error("App configuration is incomplete. Please check your secrets.")
    else:
        try:
            # Step 1: Generate SQL
            with st.spinner("Generating SQL..."):
                sql = nl_to_sql(question)

            st.subheader("Generated SQL")
            st.code(sql, language="sql")

            # Step 2: Run query
            with st.spinner("Running query on BigQuery..."):
                df = run_query(sql)

            st.subheader("Query Result")
            st.dataframe(df, use_container_width=True)

            # Step 3: Generate GPT insight
            with st.spinner("Generating insight..."):
                prompt = build_prompt(df, question)
                raw_output = generate_insight(prompt)
                parsed = parse_insight(raw_output)

            st.subheader("Summary")
            st.write(parsed["summary"] if parsed["summary"] else "No summary generated.")

            st.subheader("Insight")
            st.write(parsed["insight"] if parsed["insight"] else "No insight generated.")

            st.subheader("Recommendation")
            st.write(parsed["recommendation"] if parsed["recommendation"] else "No recommendation generated.")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# Optional Power BI section
# =========================
st.markdown("---")
st.subheader("Power BI Dashboard")

try:
    power_bi_url = st.secrets.get("POWER_BI_EMBED_URL", "")
except Exception:
    power_bi_url = ""

if power_bi_url:
    components.iframe(power_bi_url, height=600, scrolling=True)
else:
    st.info("Add POWER_BI_EMBED_URL to Streamlit secrets if you want to display the dashboard.")