import streamlit as st


st.page_link("app.py", label="Home")

data = st.file_uploader("Upload a file")
st.write(data)


name = st.text_input("First name")
st.write(f"Hello {name}")

choice = st.number_input("Pick a number", 0,100)
text = st.text_area("Text to translate")

if st.button("Say hello"):
    st.write("hello world!")
else:
    st.write("Bye")

text_contents = """ This is xxx """
