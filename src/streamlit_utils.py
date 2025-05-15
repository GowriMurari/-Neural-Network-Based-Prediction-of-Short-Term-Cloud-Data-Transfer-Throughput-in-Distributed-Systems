import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

class DataContent:
    """Class to store project markdown descriptions."""

    problem_statement = """
    ## 🏥 Problem Statement

    Modern cloud computing platforms manage immense volumes of data traffic across geographically dispersed infrastructures. Predicting the short-term variations in cloud data transfer throughput is vital for resource optimization tasks such as load balancing, replica selection, and auto-scaling. However, due to the dynamic nature of cloud environments, this task becomes extremely challenging. Traditional models either oversimplify the network behaviors or fail to incorporate key parameters like disk I/O and instance-level metrics. This project aims to bridge that gap using a neural network-based approach for intelligent throughput forecasting. 

    ### 🎯 Project Objective 
    
    The main goal of this project is to develop robust multivariate neural network models to forecast one-step-ahead throughput in cloud environments using real-time, multi-source metrics. By leveraging monitoring data from both the source and destination systems in AWS, the models can predict both network throughput (NetworkIn) and disk throughput (VolumeWriteBytes) with high accuracy. This prediction can empower cloud service providers and users with actionable insights to improve performance consistency, enhance quality-of-service, and reduce unexpected delays in large-scale data transfers.

    ### 🧠 Neural Network Models Used
    To tackle the complexity of throughput prediction, the study implemented and benchmarked seven advanced deep learning architectures:
    🔹 MLP (Multi-layer Perceptron)

    🔹 CNN (Convolutional Neural Network)
    
    🔹 RNN (Recurrent Neural Network)
    
    🔹 LSTM (Long Short-Term Memory)
    
    🔹 CNN-RNN Hybrid
    
    🔹 CNN-LSTM Hybrid
    
    🔹 Transformer Network

    Each model was rigorously optimized using grid search over hyperparameters such as number of time steps, layers, neurons, learning rate, kernel types, and more. Notably, CNN and RNN outperformed others, achieving MAPE of 3.73% (Network) and 6.16% (Disk) respectively—setting a new benchmark in short-term cloud throughput prediction.   

     ### 📊 Model Evaluation Metrics
    - Mean Absolute Percentage Error (MAPE)
    - Multivariate vs Univariate Comparison
    - Effect of file size, latency, and instance type on model accuracy
    - Transfer Learning Performance

    ### 🌐 Web-Based Implementation
    The solution is deployed using **Streamlit**, offering an interactive interface for users to:
    - Upload metrics
    - Run predictions in real-time
    - Visualize metric trends and prediction accuracy

    🏆 This project showcases how deep learning can power **intelligent cloud infrastructure decisions** through accurate, fine-grained throughput predictions.
    """

    project_data_details = """
    ### 📂 Project Data Description

    ### 🔍 Context
    The dataset was created by conducting over **5 TB** of real file transfers between AWS S3 storage and EC2 instances across **27 different configurations**, varying in **file size, instance type, and region**. Unlike traditional datasets, it includes **multivariate time series metrics** that reflect real-world network and system behaviors.

    ### 📄 Content
    The following metrics were collected at **1-minute intervals** using AWS CloudWatch:
    
    - **NetworkIn (NI):** Incoming network traffic on EC2 instance
    - **VolumeWriteBytes (VWB):** Disk throughput at the destination
    - **CPUUtilization (CU):** Processor usage on EC2 instance
    - **BytesDownloaded (BD):** Source-side throughput from S3
    - **TotalRequestLatency (TRL):** Latency between request and full response from S3

    These metrics were collected using an automated Python workflow and stored in JSON format in DynamoDB for transformation and modeling.

    ### 📉 Dataset Characteristics
    - Sampling Rate: 1-minute granularity
    - File Sizes: 1MB, 100MB, 1GB
    - Instance Types: t2.small, t3.small, m5.large
    - Source Regions: us-east-1, us-west-1, etc.
    - Dataset Size: 5 TB+ of transfer logs

    This dataset provides the first **public multivariate time series dataset** from cloud transfers including end-system parameters, enabling powerful predictive modeling.
    """

    Data_preprocessing ="""
    ### 🛠️ Data Preprocessing Steps

    ### 🔧 1. Metric Collection & Alignment
    Metrics were collected from AWS CloudWatch at 1-minute intervals during real file transfers. To ensure consistency, all data points from different services (EC2, S3, EBS) were matched by timestamp. Any mismatched or missing timestamp entries were discarded.

    ### 📊 2. Unit Normalization & Transformation
    Raw metric units (e.g., bytes, milliseconds, percentages) were standardized:
    - Bytes → Megabits per second (Mbps)
    - Milliseconds → Seconds
    - CPU utilization was scaled using the number of vCPUs per instance to account for differences in hardware capacity.

    ### ⚖️ 3. Sequence Generation for Time Series
    The time series was segmented using a rolling window approach:
    - For each instance, `n` historical time steps were used as input to predict the next.
    - This was done for each dataset transfer individually.
    - This resulted in thousands of multivariate sequences with corresponding targets.

    ### 📝 4. Feature Scaling
    All metric values were scaled using **Min-Max Scaling** to fit within the [0, 1] range. This is essential for faster convergence and consistent learning across deep learning models.

    ### 📊 5. Train-Validation-Test Split
    The dataset was divided into 70% training, 15% validation, and 15% testing sets. The split was performed **chronologically** to maintain the time series structure and avoid data leakage.

    These steps ensure that the multivariate time series dataset is cleaned, aligned, and normalized for accurate and efficient prediction using deep neural networks.
    """
    
    ml_models="""
    ## 🚀 Machine Learning Models for Cloud Throughput Prediction

    To capture temporal and multivariate patterns from AWS cloud performance metrics, we implemented and evaluated the following deep learning models:

    ---

    ### 1️⃣ **Multilayer Perceptron (MLP)**
    A basic feedforward neural network that captures nonlinear relationships across features but lacks temporal modeling capability.

    🔹 **Default Model Performance:**
    Baseline accuracy with limited sequence awareness.

    🔹 **Hyperparameter Tuning:**
    - Number of Dense Layers: 3
    - Units per Layer: 128
    - Activation: ReLU
    - Optimizer: Adam
    - Learning Rate: 0.001

    **Best MAPE:** `10.14% (NetworkIn)`

    ---

    ### 2️⃣ **Convolutional Neural Network (CNN)**
    Extracts local temporal features using convolutional filters. Effective in identifying short-term bursts and throughput spikes.

    🔹 **Default Model Performance:**
    Performs well on sequence-based features but limited context.

    🔹 **Hyperparameter Tuning:**
    - Conv Layers: 2
    - Filters: 64
    - Kernel Size: 3
    - Pooling: MaxPooling (size=2)
    - Dropout: 0.3

    **Best MAPE:** `3.73% (NetworkIn)` ✅

    ---

    ### 3️⃣ **Recurrent Neural Network (RNN)**
    Designed for sequential learning. Handles short dependencies well but struggles with longer ones.

    🔹 **Default Model Performance:**
    Decent, but highly sensitive to time-step tuning.

    🔹 **Hyperparameter Tuning:**
    - RNN Units: 64
    - Dropout: 0.2
    - Time Steps: 11
    - Optimizer: RMSprop

    **Best MAPE:** `6.16% (VolumeWriteBytes)` ✅

    ---

    ### 4️⃣ **Long Short-Term Memory (LSTM)**
    An improved RNN that remembers longer temporal patterns. Suitable for bursty or delayed effects.

    🔹 **Default Model Performance:**
    Better than basic RNNs, but slower to train.

    🔹 **Hyperparameter Tuning:**
    - LSTM Units: 64–128
    - Dropout: 0.3
    - Recurrent Dropout: 0.2
    - Learning Rate: 0.0005

    **Best MAPE:** `6.71%`

    ---

    ### 5️⃣ **CNN-RNN Hybrid**
    Combines spatial pattern extraction of CNN with sequence modeling of RNN.

    🔹 **Architecture:**
    - CNN Layer → Pooling → RNN Layer → Dense
    - Captures both local and sequential dependencies.

    **Best MAPE:** `4.21%`

    ---

    ### 6️⃣ **CNN-LSTM Hybrid**
    Uses LSTM in place of RNN for richer sequential context.

    🔹 **Best Parameters:**
    - CNN Filters: 64
    - LSTM Units: 128
    - Dropout: 0.4

    **Best MAPE:** `4.48%`

    ---

    ### 7️⃣ **Transformer Network**
    Leverages attention to weigh importance of time steps. Good for long-range dependency modeling but may lose temporal order.

    🔹 **Performance:**
    Underperformed on short-term 1-step predictions due to permutation invariance.

    🔹 **Best Parameters:**
    - Heads: 4
    - Layers: 2
    - Feed-forward Dim: 256

    **Best MAPE:** `8.53%`

    ---

    These models were benchmarked on the same dataset, and **CNN for NetworkIn** and **RNN for VolumeWriteBytes** emerged as the top performers based on MAPE.
    """
        
class DataTable:
    """Class to handle dataset loading and displaying with AgGrid."""
    
    def __init__(self, df):
        self.df = df

    def display_table(self):
        df_preview = self.df.head(100)

        gb = GridOptionsBuilder.from_dataframe(df_preview)
        gb.configure_default_column(
            groupable=True,
            value=True,
            enableRowGroup=True,
            editable=False
        )

        # Custom Styling
        gb.configure_grid_options(
            rowHeight=40,
            headerHeight=50,
            domLayout="autoHeight",
            suppressHorizontalScroll=True,
            enableSorting=True,
            enableFilter=True,
            rowSelection='multiple',
        )

        grid_options = gb.build()

        custom_css = {
            ".ag-header": {  
                "background-color": "#0047AB",
                "color": "#FFFFFF",
                "font-size": "16px",
                "font-weight": "bold",
                "text-align": "center",
                "border-bottom": "2px solid #CCCCCC",
                "padding": "10px"
            },
            ".ag-header-cell": {
                "background-color": "#0047AB !important",
                "color": "#FFFFFF !important",
                "border": "none",
                "padding": "5px",
                "height": "50px",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
            },
            ".ag-row-odd": {
                "background-color": "#F8F9FA",
            },
            ".ag-row-even": {
                "background-color": "#E9ECEF",
            },
            ".ag-body": {
                "border": "2px solid #CCCCCC",
            },
            ".ag-cell": {
                "font-size": "14px",
                "color": "#333333",
            }
        }

        # Apply AgGrid with Custom Styling
        AgGrid(
            df_preview,
            gridOptions=grid_options,
            enable_enterprise_modules=True,
            theme="balham",
            height=600,
            custom_css=custom_css
        )
