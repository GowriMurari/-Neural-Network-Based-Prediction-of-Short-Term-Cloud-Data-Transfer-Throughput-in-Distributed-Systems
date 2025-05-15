import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataContent:
    """Class to store project markdown descriptions."""

    # (All previous content remains unchanged...)

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


class DataVisualizer:
    def __init__(self):
        self.layout = {
            'template': 'plotly_white',
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'black',
            'font': dict(color='white')
        }

    def plot_correlation_heatmap(self, data):
        correlation_matrix = data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlation'),
            text=correlation_matrix.round(2).astype(str).values, 
            texttemplate="%{text}",  
            hoverinfo="text" 
        ))

        fig.update_layout(
            **self.layout,
            title='Correlation Heatmap of Cloud Metrics',
            title_x=0.4,
            xaxis_title='Features',
            yaxis_title='Features',
            width=1000,
            height=800,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig

    def plot_metric_scatter(self, data, x_metric, y_metric):
        fig = px.scatter(
            data,
            x=x_metric,
            y=y_metric,
            color='SourceRegion' if 'SourceRegion' in data.columns else None,
            title=f"Time Series Scatter of {y_metric} by {x_metric}",
            template="plotly_white"
        )
        fig.update_layout(
            **self.layout,
            title_x=0.4,
            xaxis_title=x_metric,
            yaxis_title=y_metric,
            width=1000,
            height=600
        )
        return fig