import streamlit as st
import requests

st.title("Rag Novel Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the text ...."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = requests.post(
            "http://localhost:8000/chat",
            json={"question": prompt}
        )
        answer = response.json()["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
