import streamlit as st
from supervisor_agent.graph import supervisor_graph
from pathlib import Path
import tempfile
import uuid
from dotenv import load_dotenv
import os

load_dotenv()

# st.write("Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))
# st.write("Project:", os.getenv("LANGCHAIN_PROJECT"))
# st.write("API key present:", bool(os.getenv("LANGCHAIN_API_KEY")))

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

st.set_page_config(
    page_title="raghuvar",
    page_icon="",
    layout="centered",
)

st.title("Welcome to my Agentic RAG app")

# UI chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input + attachments
prompt = st.chat_input(
    "Type your message...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp", "pdf"],
)

# Handle submission
if prompt:
    text = prompt.text
    files = prompt.files
    # Display user message
    with st.chat_message("user"):
        if text:
            st.markdown(text)

        for file in files:
            st.caption(f"📎 {file.name}")

    # Save text to UI history
    if text:
        st.session_state.messages.append({
            "role": "user",
            "content": text,
        })

    # Prepare input for backend
    if files:
        uploaded_file = files[0]
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:
            tmp.write(uploaded_file.getbuffer())
            file_path = Path(tmp.name)
        raw_input = file_path
    else:
        raw_input = text

    response = supervisor_graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": text if text else f"Uploaded file: {files[0].name}"
                }
            ],
            "raw_input": raw_input,
        },
        config=config
    )

    final_response = response["final_response"]
    # Display response
    with st.chat_message("assistant"):
        st.markdown(final_response)

    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_response,
    })
