import streamlit as st

# JavaScript to modify the viewport meta tag
viewport_script = """
<script>
document.querySelector('meta[name="viewport"]').setAttribute("content", "width=device-width, initial-scale=1");
</script>
"""

# Inject the JavaScript into the Streamlit app
st.markdown(viewport_script, unsafe_allow_html=True)

# Rest of your Streamlit code follows...
