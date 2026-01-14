import streamlit as st
import pandas as pd
from prophet import Prophet
import datetime
import os
import matplotlib.pyplot as plt
import google.generativeai as genai

# ==========================
# CONFIGURE GEMINI API
# ==========================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ==========================
# OUTPUT DIRECTORY
# ==========================
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# STREAMLIT APP UI
# ==========================
st.title("📈 ReAmp TicketNext")

uploaded_file = st.file_uploader(
    "Upload your data file (CSV or Excel)",
    type=["csv", "xlsx", "xls"]
)

required_columns = ["CREATED_ON"]

if uploaded_file:
    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        else:
            st.error("Unsupported file format.")
            st.stop()

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()


    # Ensure datetime conversion
    if "CREATED_ON" in df.columns:
        df["CREATED_ON"] = pd.to_datetime(df["CREATED_ON"], errors="coerce")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required column(s): {', '.join(missing_cols)}")

    else:
        # Feature selection
        feature = st.radio(
            "Choose a Feature",
            ["📈 Forecast Tickets", "🤖 AI-powered Data Assistant"]
        )

        # =====================================
        # 📈 FORECAST FEATURE
        # =====================================
        if feature == "📈 Forecast Tickets Using Prophet":

            forecast_days = st.slider("Select Forecast Horizon (Days)", 30, 365, 180, step=30)
            df = df.dropna(subset=["CREATED_ON"])

            daily = df.set_index("CREATED_ON").resample("D").size().to_frame("y").reset_index()
            daily.rename(columns={"CREATED_ON": "ds"}, inplace=True)

            if len(daily) < 2:
                st.warning("Not enough data points to forecast.")
            else:
                st.subheader("🔮 Forecasting with Prophet")

                with st.spinner("Running forecast..."):
                    m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
                    m.fit(daily)
                    future = m.make_future_dataframe(periods=forecast_days)
                    forecast = m.predict(future)

                forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[
                    ["yhat", "yhat_lower", "yhat_upper"]
                ].clip(lower=0)

                last_actual_date = daily["ds"].max()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                output_file = os.path.join(
                    OUTPUT_DIR,
                    f"forecast_daily_{timestamp}.csv"
                )
                forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(output_file, index=False)

                st.success(f"✅ Forecast saved at: {output_file}")

                # Plot Forecast
                st.subheader("📊 Forecast Plot")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(daily["ds"], daily["y"], "o", label="Actual Tickets")
                ax.plot(forecast["ds"], forecast["yhat"], "-", label="Forecast")
                ax.fill_between(
                    forecast["ds"],
                    forecast["yhat_lower"],
                    forecast["yhat_upper"],
                    alpha=0.3,
                    label="Confidence Interval"
                )
                ax.axvspan(last_actual_date, forecast["ds"].max(), color="lightgrey", alpha=0.2)
                ax.legend()
                st.pyplot(fig)

        # =====================================
        # 🤖 GEMINI AI DATA ASSISTANT (UPGRADED)
        # =====================================
        elif feature == "🤖 Data Q&A Using gemini":

            st.subheader("💬 Talk to Your Data")

            query = st.text_area("Ask anything about your data:")

            if st.button("Run Query"):
                if not query.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Thinking..."):

                        # Master prompt = analytics + natural language
                        prompt = f"""
                        You are an expert data analyst. The user uploaded a pandas DataFrame named df.

                        Columns available:
                        {list(df.columns)}

                        User question:
                        "{query}"

                        Your job:
                        1. If the question requires data analysis, generate pandas Python code to answer it.
                        2. If it requires explanation, provide a natural language summary.
                        3. If both are required, provide explanation + code.
                        4. Code must store the final answer in a variable named "result".

                        Return in this format:

                        RESPONSE:
                        <Your natural-language explanation>

                        CODE:
                        ```python
                        # your pandas code here
                        result = ...
                        ```
                        """

                        model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
                        response = model.generate_content(prompt)

                        full_text = response.text

                        # Extract the natural language part
                        explanation = full_text.split("CODE:")[0].replace("RESPONSE:", "").strip()

                        # Extract the code
                        code_block = ""
                        if "```python" in full_text:
                            code_block = full_text.split("```python")[1].split("```")[0].strip()

                        # Show natural-language insight
                        st.markdown("### 🧠 AI Insight")
                        st.write(explanation)

                        # Show Python code
                        if code_block:
                            st.markdown("### 🧪 Generated Code")
                            st.code(code_block, language="python")

                            try:
                                local_vars = {"df": df, "pd": pd}
                                exec(code_block, {}, local_vars)

                                # Show result
                                if "result" in local_vars:
                                    st.markdown("### 📊 Result")
                                    st.write(local_vars["result"])

                            except Exception as e:
                                st.error(f"⚠️ Error executing the code: {e}")
