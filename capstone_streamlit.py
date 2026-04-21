import streamlit as st
import uuid
from agent import get_llm, setup_kb, create_graph

st.set_page_config(page_title="E-Commerce FAQ Bot", page_icon="🛒", layout="wide")

def init_agent():
    # No st.* calls inside here — they trigger re-renders mid-init
    llm = get_llm()
    embedder, collection = setup_kb()
    app = create_graph(llm, embedder, collection)
    return app

if "agent_app" not in st.session_state:
    try:
        with st.spinner("🦙 Starting Qwen2.5:3b via Ollama..."):
            st.session_state.agent_app = init_agent()
    except Exception as e:
        st.error(f"⚠️ Failed to start Ollama. Is it running? Error: {e}")
        import traceback
        traceback.print_exc()
        st.stop()

app = st.session_state.agent_app

# Initialize session state variables
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi there! I'm your intelligent E-Commerce assistant. How can I help you today?"}
    ]

with st.sidebar:
    st.title("E-Commerce FAQ Bot 🛒")
    st.write("Welcome! This assistant is equipped to answer queries about:")
    st.markdown("""
    - Returns & Cancellations
    - Shipping Times & Types
    - Tracking Missing/Damaged items
    - Gift Cards & Payments
    """)
    st.divider()
    if st.button("New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = [
            {"role": "assistant", "content": "New conversation started! How can I help you today?"}
        ]
        st.rerun()

st.title("Customer Support Chat")

# Display history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about returns, shipping, or check the time...")

if user_input:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = app.invoke({"question": user_input}, config=config)
            
            # The answer is stored in result["answer"]
            ans = result.get("answer", "Sorry, I am unable to answer that right now.")
            st.markdown(ans)
    
    # Save to history
    st.session_state.chat_history.append({"role": "assistant", "content": ans})
