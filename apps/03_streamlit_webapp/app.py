import streamlit as st

st.set_page_config(page_title="YOLO Streamlit Dashboard", layout="wide")
st.title("YOLO Streamlit Dashboard")
st.write("Bienvenido al dashboard de visión por computador con YOLO.")

st.info("Sube una imagen y visualiza los resultados de detección aquí (demo placeholder)")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Imagen subida", use_column_width=True)
    st.success("Aquí se mostrarían los resultados de YOLO (demo)")
