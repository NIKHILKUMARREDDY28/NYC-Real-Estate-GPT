import streamlit as st


# Streamlit UI
st.title("QA Application with OpenAI and Llama Models")

# Model selection
model_choice = st.selectbox("Choose a model:", ["OpenAI gpt-4o", "Llama 3.1"])

if model_choice == "OpenAI gpt-4o" :
    # User input for OpenAI API key
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key (optional):", type="password")
    if not openai_api_key:
        st.error("OPENAI KEY IS MANDATORY FOR USING OPENAI LLM")

# User input
user_input = st.text_area("Enter your question:")

if st.button("Get Answer"):
    if user_input:
        answer = ""
        st.write(f"**Answer from {model_choice}:** {answer}")
    else:
        st.write("Please enter a question.")
