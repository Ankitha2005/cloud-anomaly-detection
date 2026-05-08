"""
Minimal test app for Streamlit Cloud
"""
import streamlit as st

st.set_page_config(page_title="Test App", page_icon="✅")

st.title("✅ Streamlit Cloud Test")
st.write("If you can see this, the deployment is working!")
st.success("All dependencies loaded successfully!")

# Test imports
try:
    import pandas as pd
    import numpy as np
    import plotly.express as px
    
    st.write("✅ pandas:", pd.__version__)
    st.write("✅ numpy:", np.__version__)
    st.write("✅ plotly:", px.__version__)
    
    # Simple chart
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })
    
    fig = px.line(df, x='x', y='y', title='Simple Test Chart')
    st.plotly_chart(fig)
    
except Exception as e:
    st.error(f"Error: {e}")
