import streamlit as st
import uuid
from bot import get_llm, setup_kb, create_graph

st.set_page_config(page_title="E-Commerce FAQ Bot", page_icon="🛒", layout="wide")

def init_agent():
    llm = get_llm()
    emb, col = setup_kb()
    return create_graph(llm, emb, col)

if "agent_app" not in st.session_state:
    with st.spinner("Initializing Agent..."):
        st.session_state.agent_app = init_agent()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []

st.title("🛒 E-Commerce Customer Support")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about returns, shipping, payments..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = st.session_state.agent_app.invoke({"question": prompt}, config=config)
    answer = result.get("answer", "Sorry, I couldn't find an answer.")
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()
