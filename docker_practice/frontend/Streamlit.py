import streamlit as st
import requests as request
API_URL = "http://127.0.0.1:8000/forecast"
st.title("NEXYGEN Emissions Forecasting")
st.write("This app allows you to forecast Scope 1 and Scope 2 emissions for NEXYGEN based on pre-trained SARIMAX models.")
with st.form("forecast_form"):
    st.subheader("Select Emission Type and Forecast Steps")
    col1, col2 = st.columns(2)
    with col1:
        emission_type = st.selectbox("Emission Type", options=["scope1", "scope2"])
        steps = st.number_input("Number of Steps to Forecast", min_value=1, max_value=365, value=30, help="Enter the number of future time steps you want to forecast (e.g., 30 for next 30 days).")
        submitted = st.form_submit_button("Forecast")
    if submitted:
        payload = {
            "emission_type": emission_type,
            "steps": steps
        }
        try:
            response = request.post(API_URL, json=payload, timeout=20)
            response.raise_for_status()  # Raise an error for bad status codes
            if response.status_code == 200:
                result = response.json()
                forecast = result['forecast']
                dates = result['dates']
                st.write(f"Forecasted {result['emission_type'].capitalize()} Emissions for the next {steps} steps:")
                for d, forecast_value in zip(dates, forecast):
                    st.write(f"{d}: {forecast_value:.2f} Metric tons of CO2e")
                st.subheader(f"Forecasted {result['emission_type'].capitalize()} Emissions")
                
                # debug output to inspect received dates
                st.write("Dates returned by API:", dates)
                
                # Create a DataFrame for the chart
                import pandas as pd
                import matplotlib.pyplot as plt
                
                chart_data = pd.DataFrame({
                    'Date': pd.to_datetime(dates),
                    'Forecast': forecast
                })
                st.write("Chart data:", chart_data)
                
                # plot with explicit ticks to avoid collapsing
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(chart_data['Date'], chart_data['Forecast'], marker='o')
                ax.set_title(f"Forecasted {result['emission_type'].capitalize()} Emissions")
                ax.set_xlabel("Date")
                ax.set_ylabel("Emissions (Metric tons of CO2e)")
                ax.set_xticks(chart_data['Date'])
                ax.set_xticklabels(chart_data['Date'].dt.strftime('%Y-%m-%d'), rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except request.exceptions.ConnectionError:
            st.error("Connection error: Unable to connect to the API. Please ensure the API server is running.")    
        except request.exceptions.Timeout:
            st.error("Timeout error: The request to the API timed out. Please try again later.")
        except request.exceptions.HTTPError as e:
            st.error(f"Backend returned an error: {e.response.text}")
        except Exception as e:
            st.error(f"An Unexpected error occurred: {e}")
