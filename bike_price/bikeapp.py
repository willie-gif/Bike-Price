import streamlit as st
import pickle
import numpy as np
import pandas as pd
from fpdf import FPDF
import io
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu 


# Page setup
st.set_page_config(page_title="Bike Prices", layout="wide", page_icon="🚴")

# Load model and scaler
def load_model():
    with open("bike.pkl", "rb") as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data["regressor"]
scaler = data["scaler"]

# Initialize session state to store predictions
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Date", "Product Name", "Quantity", "Unit Cost", "Profit", "Total Cost", "Revenue", "Predicted Price"])

# Sidebar
with st.sidebar:
    selected = option_menu(
        "Bike Price Prediction System",
        ["Home", "Details","Report"],
        icons=["house", "pencil", "bar-chart"],
        menu_icon="🚴",
        default_index=0
    )

# HOME SECTION
if selected == "Home":
    st.title("🏠 Home")
    st.write("Welcome to the Bike Price Prediction System.")
    



# REPORT SECTION
if selected == "Report":
    st.subheader("📅 Generate Report")
    report_type = st.selectbox("Select report type", ["Daily", "Weekly", "Monthly", "Yearly"])

    if st.button("Generate Report"):
        df = st.session_state.data.copy()
        now = datetime.now()

        # Ensure date column is datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter by report type
        if report_type == "Daily":
            filtered = df[df['Date'].dt.date == now.date()]
        elif report_type == "Weekly":
            week_ago = now - timedelta(days=7)
            filtered = df[df['Date'] >= week_ago]
        elif report_type == "Monthly":
            month_start = now.replace(day=1)
            filtered = df[df['Date'] >= month_start]
        elif report_type == "Yearly":
            year_start = now.replace(month=1, day=1)
            filtered = df[df['Date'] >= year_start]

        st.write(f"📈 {report_type} Report")
        st.dataframe(filtered)

        # Excel download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            filtered.to_excel(writer, index=False, sheet_name="Report")
        st.download_button(
            label="📥 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"{report_type.lower()}_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # PDF download
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"{report_type} Report", ln=True, align='C')

        for _, row in filtered.iterrows():
            row_str = f"Date: {row['Date']} | Product Name: {row['Product Name']} | Qty: {row['Quantity']} | Unit Cost : ${row['Unit Cost']} | Profit : ${row['Profit']} | Total Cost : ${row['Total Cost']} | Revenue : ${row['Revenue']} | Price: ${row['Predicted Price']:.2f}"
            pdf.cell(200, 10, txt=row_str, ln=True)

        # ✅ Fixed PDF output
        pdf_buffer = io.BytesIO()
        pdf_output = pdf.output(dest='S').encode('latin-1')  # Returns bytes
        pdf_buffer.write(pdf_output)

        st.download_button(
            label="📥 Download PDF",
            data=pdf_buffer.getvalue(),
            file_name=f"{report_type.lower()}_report.pdf",
            mime="application/pdf"
        )


# DETAILS SECTION
elif selected == "Details":
    st.title("Product Details")
    st.write("Please Fill Out The Following Details:")

    col1, col2 = st.columns(2)

    with col1:
        product_name = st.text_input("Product Name")
        profit = st.number_input("Profit Amount", value=None)
        revenue = st.number_input("Revenue Amount", value=None)
    with col2:
        unit_cost = st.number_input("Unit Cost", value=None)
        total_cost = st.number_input("Total Cost", value=None)
        Quantity = st.slider("Quantity", 1, 100, 1)

    if st.button("Predict"):
        x_new = np.array([[Quantity, unit_cost, profit, total_cost, revenue]])
        x_new = x_new.astype(float)
        x_new = scaler.transform(x_new)

        bike = model.predict(x_new)
        st.subheader(f"The Unit Price should be: ${bike[0]:.2f}")

        # Store the prediction with inputs and timestamp
        new_row = {
            "Date": datetime.now(),
            "Product Name": product_name,
            "Quantity": Quantity,
            "Unit Cost": unit_cost,
            "Profit": profit,
            "Total Cost": total_cost,
            "Revenue": revenue,
            "Predicted Price": bike[0]
        }
        st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True)


