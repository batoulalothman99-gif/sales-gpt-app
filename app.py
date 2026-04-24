import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
from google.cloud import bigquery
from google.oauth2 import service_account

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="AI Sales Insights Dashboard",
    layout="wide",
    page_icon="📊"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f2937;
}
.sub-title {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 25px;
}
.card {
    padding: 20px;
    border-radius: 16px;
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
}
.metric-card {
    padding: 18px;
    border-radius: 14px;
    background-color: #eef2ff;
    border: 1px solid #c7d2fe;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown('<div class="main-title">AI Sales Insights Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask questions about your sales data (2015-2018) and get instant SQL, results, insights, and recommendations.</div>',
    unsafe_allow_html=True
)

# =========================
# Constants
# =========================
PROJECT_ID = "project1-488217"
DATASET_ID = "dw_sales"

# =========================
# Functions: Setup
# =========================
@st.cache_resource
def get_gemini_model():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
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
        st.error(f"BigQuery configuration error: {e}")
        return None


model = get_gemini_model()
client = get_bigquery_client()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("System Status")
    st.write("Gemini:", "✅ Ready" if model else "❌ Not Ready")
    st.write("BigQuery:", "✅ Ready" if client else "❌ Not Ready")

    st.markdown("---")
    st.subheader("Sample Questions")
    st.write("• What is the total sales?")
    st.write("• What is the average sales in 2017?")
    st.write("• Show sales by region")
    st.write("• What are the top 5 products by sales?")
    st.write("• Show sales by customer segment")

# =========================
# Schema context
# =========================
DW_SCHEMA_CONTEXT = f"""
You are a SQL expert for Google BigQuery.

Project: {PROJECT_ID}
Dataset: {DATASET_ID}

Available tables:

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

Important rules:
- Use BigQuery SQL only.
- Always use fully qualified table names.
- Use backticks for column names with spaces.
- Return only valid SQL.
- Do not return markdown.
- Do not explain the SQL.
- Avoid unnecessary joins.
- If the requested column exists in the fact table, use it directly.
- For year filtering, prefer EXTRACT(YEAR FROM DATE(`Order Date`)).
- Only join dim_date if the question explicitly requires date attributes such as quarter, month, or day.
- Join dim_customer using `Customer ID`.
- Join dim_product using `Product ID`.
- Join dim_location using `LocationKey`.
"""

# =========================
# Helper functions
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


def nl_to_sql(question: str) -> str:
    if model is None:
        raise ValueError("Gemini model is not initialized.")

    prompt = f"""
{DW_SCHEMA_CONTEXT}

Convert the following user question into a valid BigQuery SQL query.

Question:
{question}

Return only SQL.
"""

    response = model.generate_content(prompt)

    if not hasattr(response, "text") or not response.text:
        raise ValueError("Gemini did not return SQL.")

    return clean_sql(response.text)


def run_query(sql: str) -> pd.DataFrame:
    if client is None:
        raise ValueError("BigQuery client is not initialized.")

    query_job = client.query(sql)
    return query_job.to_dataframe()


def build_insight_prompt(df: pd.DataFrame, question: str) -> str:
    if df.empty:
        result_text = "The query returned no rows."
    else:
        result_text = df.head(20).to_markdown(index=False)

    return f"""
You are a business analyst.

The user asked:
{question}

The SQL result is:
{result_text}

Please answer in this exact format:

Summary:
[2-3 sentences]

Insight:
[2-3 sentences]

Recommendation:
[2-3 sentences]
"""


def generate_insight(prompt: str, retry_count: int = 3) -> str:
    if model is None:
        raise ValueError("Gemini model is not initialized.")

    for attempt in range(retry_count):
        try:
            response = model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception:
            if attempt < retry_count - 1:
                time.sleep(2)
            else:
                raise

    return ""


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
# Main UI
# =========================
st.markdown("---")

left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.subheader("Ask AI About Sales Data")

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: What is the average sales in 2017?"
    )

    submit = st.button("Get Insights 🚀", use_container_width=True)

with right_col:
    st.subheader("How it works")
    st.markdown("""
    <div class="card">
    1. You ask a business question.<br>
    2. Gemini converts it into SQL.<br>
    3. BigQuery runs the query on the data warehouse.<br>
    4. Gemini explains the result and gives recommendations.
    </div>
    """, unsafe_allow_html=True)

# =========================
# Processing
# =========================
if submit:
    if not question.strip():
        st.warning("Please enter a question.")
    elif model is None or client is None:
        st.error("App configuration is incomplete. Please check Streamlit secrets.")
    else:
        try:
            with st.spinner("Generating SQL using Gemini..."):
                sql = nl_to_sql(question)

            st.markdown("## Generated SQL")
            st.code(sql, language="sql")

            with st.spinner("Running query on BigQuery..."):
                df = run_query(sql)

            st.markdown("## Query Result")
            st.dataframe(df, use_container_width=True)

            with st.spinner("Generating business insights..."):
                insight_prompt = build_insight_prompt(df, question)
                raw_output = generate_insight(insight_prompt)
                parsed = parse_insight(raw_output)

            st.markdown("## AI Business Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Summary")
                st.success(parsed["summary"] if parsed["summary"] else "No summary generated.")

            with col2:
                st.markdown("### Insight")
                st.info(parsed["insight"] if parsed["insight"] else "No insight generated.")

            with col3:
                st.markdown("### Recommendation")
                st.warning(parsed["recommendation"] if parsed["recommendation"] else "No recommendation generated.")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# Power BI Dashboard
# =========================
st.markdown("---")
st.markdown("## Power BI Dashboard")

try:
    power_bi_url = st.secrets.get("POWER_BI_EMBED_URL", "")
except Exception:
    power_bi_url = ""

if power_bi_url:
    components.iframe(power_bi_url, height=650, scrolling=True)
else:
    st.info("Add POWER_BI_EMBED_URL to Streamlit secrets if you want to display the dashboard.")