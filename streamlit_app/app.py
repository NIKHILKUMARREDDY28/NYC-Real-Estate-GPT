import os
import sys
import traceback
import streamlit as st

# Make sure your local modules are discoverable
sys.path.append(os.getcwd())

# Local modules
from streamlit_app.data_ingestor import ChromaDB
from streamlit_app.llms.clients import OpenAILLMSClient, OllamaLLMSClient
from streamlit_app.config import settings

# -------------
# Streamlit UI
# -------------
st.title("NYC Real Estate Chatbot")

# Model selection
model_choice = st.selectbox("Choose a model:", ["OpenAI GPT-4"])

# Initialize ChromaDB
db = ChromaDB(collection_name="ACRIS", persist_directory=".chromadb")

# Initialize the conversation in session state if not present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing conversation (chat history)
for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        # assistant/system messages
        with st.chat_message("assistant"):
            st.write(message["content"])

# Prompt the user in a chat-style input (introduced in Streamlit 1.25.0)
user_input = st.chat_input("Ask a question about NYC real estate...")

# When the user enters a question
if user_input:
    # 1) Show the user's message in the chat
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    try:
        # 2) Retrieve top k documents from ChromaDB
        #    We retrieve them as function-call style content or anything that suits your LLM
        search_results = db.search_document(user_input, k=3)

        # Option A: Put the retrieved documents in a system/context message
        # You can refine how you pass these docs to your LLM:
        context_message = {
            "role": "system",
            "content": f"Relevant NYC Real Estate Documents: {search_results}"
        }
        st.session_state["messages"].append(context_message)

        # 3) Choose the LLM based on user selection
        if model_choice == "OpenAI GPT-4":
            client = OpenAILLMSClient()
        else:
            client = OllamaLLMSClient()

        # 4) Get the assistant's answer (RAG: conversation + retrieved docs)
        answer = client.get_answer(st.session_state["messages"])

        # 5) Display the assistant's answer in the chat
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

    except Exception as e:
        st.error(f"An error occurred: {traceback.format_exc()}")
