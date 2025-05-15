import streamlit as st
import numpy as np
import subprocess
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import io
from streamlit.components.v1 import html
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
from tensorflow.keras.models import load_model
from src.visualizations import DataVisualizer
from src.styles import TITLE_STYLE, SIDEBAR_STYLE
from src.streamlit_utils import DataContent, DataTable

# Define model scripts
models = {
    "CNN": None,
    #"CNN-LSTM": "data_transformation_cnnlstm_json.py",
    "LSTM": None,
    #"MLP": "data_transformation_mlp_json.py",
    "RNN": None,
}

st.set_page_config(
    page_title="Short-Term Variation",
    layout="wide",  # Expands content area
    initial_sidebar_state="expanded",  # Keeps sidebar open
)


def convert_df_to_csv(df):
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()

def main():
    st.markdown(TITLE_STYLE, unsafe_allow_html=True)
    st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

    st.markdown('<h1 class="styled-title">Predicting Short-Term Variations in End-to-End Cloud Data Transfer Throughput Using Neural Networks Application</h1>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-title">Select Options</div>', unsafe_allow_html=True)
    

    if 'page' not in st.session_state:
        st.session_state['page'] = "Problem Statement"

    if "df" not in st.session_state:
        st.session_state.df = None 
    
    if 'pre_df' not in st.session_state:
        st.session_state.pre_df = None
    
    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'preprocessed' not in st.session_state:
        st.session_state.preprocessed = None

    # Sidebar buttons
    if st.sidebar.button("Problem Statement"):
        st.session_state['page'] = "Problem Statement"

    if st.sidebar.button("Project Data Description"):
        st.session_state['page'] = "Project Data Description"

    if st.sidebar.button("Sample Training Data"):
        st.session_state['page'] = "Sample Training Data"

    if st.sidebar.button("Know About Data"):
        st.session_state['page'] = "Know About Data"

    if st.sidebar.button("Data Preprocessing"):
        st.session_state['page'] = "Data Preprocessing"

    if st.sidebar.button("Exploratory Data Analysis"):
        st.session_state['page'] = "Exploratory Data Analysis"

    if st.sidebar.button("Machine Learning Models Used"):
        st.session_state['page'] = "Machine Learning Models Used"

    if st.sidebar.button("Model Predictions"):
        st.session_state['page'] = "Model Predictions"

################################################################################################################

    if st.session_state['page']== "Problem Statement":
        st.image("./image.webp", width=500)
        st.markdown(DataContent.problem_statement)
    
    elif  st.session_state['page'] == "Project Data Description":
        st.markdown(DataContent.project_data_details)

    elif st.session_state['page'] == "Sample Training Data":
        st.markdown("## 📊 Training Data Preview")
        st.write("🔍 Below is an **interactive table** displaying the first 100 rows:")
        file_path = r"./data/test.csv"
        st.session_state.df = pd.read_csv(file_path)
        data_table = DataTable(df=st.session_state.df)
        data_table.display_table()


    elif  st.session_state['page'] == "Know About Data":
        file_path = r"./data/test.csv"
        st.session_state.df = pd.read_csv(file_path)
        st.header("Data Information")

        if "profile_report_generated" not in st.session_state:
            with st.status("⏳ Generating Overall Data Profile Report...", expanded=True) as status:
                profile = ProfileReport(st.session_state.df, explorative=True)
                profile.to_file("ydata_profiling_report.html")
                st.session_state["profile_report_generated"] = True  # Mark as generated
                status.update(label="✅ Report Generated Successfully!", state="complete")

        try:
            with open("ydata_profiling_report.html", "r", encoding="utf-8") as f:
                report_html = f.read()
            html(report_html, height=1000,width=800, scrolling=True)  

        except FileNotFoundError:
            st.error("Report file not found. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


    elif  st.session_state['page'] == "Data Preprocessing":
        st.markdown(DataContent.Data_preprocessing)
        pre_df_file = r"./data/test.csv"
        st.session_state.pre_df = pd.read_csv(pre_df_file)
        st.write("### Preprocessed Data Preview (First 15 Rows)")
        data_table = DataTable(df=st.session_state.pre_df.head(15))
        data_table.display_table()


    elif st.session_state['page'] == "Exploratory Data Analysis":
        file_path = r"./data/test.csv"
        st.session_state.df = pd.read_csv(file_path)
        st.header("📊 Data Visualization")
        visualizer = DataVisualizer()

        plot_type = st.selectbox(
            "Select Visualization", 
            ["Correlation Heatmap", "Metric Scatter Plot"]
        )

        if plot_type == "Correlation Heatmap":
            with st.spinner("Generating Correlation Heatmap..."):
                fig = visualizer.plot_correlation_heatmap(st.session_state.df)
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Metric Scatter Plot":
            with st.spinner("Generating Metric Scatter Plot..."):
                if "Timestamp" not in st.session_state.df.columns:
                    st.session_state.df["Timestamp"] = pd.date_range(
                        start="2023-01-01", 
                        periods=len(st.session_state.df), 
                        freq="T"
                    )

                x_col = st.selectbox("Select X-axis Metric", st.session_state.df.columns)
                y_col = st.selectbox("Select Y-axis Metric", [col for col in st.session_state.df.columns if col != x_col])
                fig = visualizer.plot_metric_scatter(st.session_state.df, x_metric=x_col, y_metric=y_col)
                st.plotly_chart(fig, use_container_width=True)

            
    elif st.session_state['page'] == "Machine Learning Models Used":
        st.markdown(DataContent.ml_models)

    elif st.session_state.get('page') == "Model Predictions":
            
        # Upload dataset file
        dataset_file = st.file_uploader("Upload Dataset JSON", type=["json"])
        test_csv_file = st.file_uploader("Upload Test Input CSV (9x5)", type=["csv"])

        sample_input = np.array([
            [120, 75, 300, 50, 200],
            [130, 78, 290, 55, 210],
            [125, 76, 295, 52, 205],
            [140, 80, 310, 60, 220],
            [135, 79, 305, 58, 215],
            [128, 77, 298, 53, 208],
            [138, 81, 315, 63, 225],
            [145, 85, 320, 65, 230],
            [150, 88, 330, 70, 240]
        ])

        if test_csv_file:
            df_test = pd.read_csv(test_csv_file)
            if df_test.shape == (9, 5):
                sample_input = df_test.to_numpy()
                st.success("Test input loaded from CSV")
            else:
                st.warning("Uploaded CSV must be exactly 9 rows × 5 columns")

        sample_input = sample_input.reshape((1, 9, 5))

        if dataset_file:
            dataset_path = "dataset.json"
            with open(dataset_path, "wb") as f:
                f.write(dataset_file.getbuffer())
            st.success("Dataset uploaded successfully!")

            with open(dataset_path, "r") as f:
                data = json.load(f)

            data_points = []
            for item in data.get("Items", []):
                for dp in item.get("DataPoints", []):
                    dp["Date"] = item["Date"]
                    data_points.append(dp)

            df = pd.DataFrame(data_points)
            st.subheader("Preview of Dataset")
            st.write(df.head())

            st.subheader("CPU Utilization Trend")
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df["CPUUtilizationSum"], marker='o', linestyle='-')
            plt.xlabel("Time")
            plt.ylabel("CPU Utilization")
            plt.title("CPU Utilization Trend Over Time")
            st.pyplot(plt)

            st.subheader("Network Traffic Trend")
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df["NetworkInSum"], marker='s', linestyle='-', color='r')
            plt.xlabel("Time")
            plt.ylabel("Network Traffic (bytes)")
            plt.title("Network Traffic Over Time")
            st.pyplot(plt)

        model_choice = st.selectbox("Select a Model", list(models.keys()))

        if st.button("Run Model"):
            if model_choice == "LSTM":
                try:
                    model = load_model("./lstm_model.h5", safe_mode=False)
                    st.success("LSTM Model loaded successfully from lstm_model.h5")
                    y_predicted = model.predict(sample_input)
                    st.subheader("Predicted Values for Next Time Step:")
                    st.write(f"Network Traffic: {y_predicted[0][0]:.2f} Mbps")
                    st.write(f"CPU Utilization: {y_predicted[0][1]:.2f} %")
                    st.write(f"Disk Write Speed: {y_predicted[0][2]:.2f} Mbps")
                    st.write(f"Bytes Downloaded: {y_predicted[0][3]:.2f} MB")
                    st.write(f"Request Latency: {y_predicted[0][4]:.2f} ms")
                except Exception as e:
                    st.error("Failed to load LSTM model: " + str(e))

            elif model_choice == "CNN":
                try:
                    model = load_model("./cnn_model.h5", safe_mode=False)
                    st.success("CNN Model loaded successfully from cnn_model.h5")
                    y_predicted = model.predict(sample_input)
                    st.subheader("Predicted Values for Next Time Step:")
                    st.write(f"Network Traffic: {y_predicted[0][0]:.2f} Mbps")
                    st.write(f"CPU Utilization: {y_predicted[0][1]:.2f} %")
                    st.write(f"Disk Write Speed: {y_predicted[0][2]:.2f} Mbps")
                    st.write(f"Bytes Downloaded: {y_predicted[0][3]:.2f} MB")
                    st.write(f"Request Latency: {y_predicted[0][4]:.2f} ms")
                except Exception as e:
                    st.error("Failed to load CNN model: " + str(e))

            elif model_choice == "RNN":
                try:
                    model = load_model("./rnn_model.h5", safe_mode=False)
                    st.success("RNN Model loaded successfully from rnn.h5")
                    y_predicted = model.predict(sample_input)
                    st.subheader("Predicted Values for Next Time Step:")
                    st.write(f"Network Traffic: {y_predicted[0][0]:.2f} Mbps")
                    st.write(f"CPU Utilization: {y_predicted[0][1]:.2f} %")
                    st.write(f"Disk Write Speed: {y_predicted[0][2]:.2f} Mbps")
                    st.write(f"Bytes Downloaded: {y_predicted[0][3]:.2f} MB")
                    st.write(f"Request Latency: {y_predicted[0][4]:.2f} ms")
                except Exception as e:
                    st.error("Failed to load RNN model: " + str(e))

            else:
                if dataset_file:
                    script_path = models[model_choice]
                    with st.spinner(f"Running {model_choice} model..."):
                        process = subprocess.run(["python", script_path], capture_output=True, text=True)
                        st.text_area("Output", process.stdout)
                        st.error(process.stderr) if process.stderr else st.success("Execution Completed!")
                else:
                    st.error("Please upload a dataset first.")

if __name__ == "__main__":
    main()