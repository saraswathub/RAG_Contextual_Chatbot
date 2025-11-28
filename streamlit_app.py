import streamlit as st
from rag_pipeline import ChatRAG

st.set_page_config(page_title="Contextual Chatbot", layout="centered")

@st.cache_resource
def get_rag():
    return ChatRAG()

rag = get_rag()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Contextual Chatbot — Local RAG")
query = st.text_input("Ask a question about the docs:")

if st.button("Send") and query:
    with st.spinner("Thinking..."):
        answer, sources = rag.answer(query)
        st.session_state.last_sources = sources
    st.session_state.history.append((query, answer))

for q,a in st.session_state.history[::-1]:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}\n---")

st.sidebar.header("Retrieved Sources")

if 'last_sources' in st.session_state and st.session_state.last_sources:
    for s in st.session_state.last_sources:
        st.markdown(f"- {s}")
else:
    st.markdown("No sources yet. Run a query to populate retrieved sources.")
