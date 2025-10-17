# app.py

import streamlit as st
import os
from hybrid_agent import hybrid_agent_execute # Import your core logic

# --- UI Setup ---
st.set_page_config(page_title="Hybrid AI Agent", page_icon="💡", layout="centered")
st.title("💡 Hybrid AI Agent (GPT + Gemini)")
st.caption("A powerful, synthesized response from two leading LLMs.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Hybrid Agent. Ask me a complex question to see my synthesis in action."}
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Enter your prompt here..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Assistant Response Container
    with st.chat_message("assistant", avatar="🤖"):
        # Show a spinner while the agent is running
        with st.spinner("🧠 Agent is consulting GPT and Gemini for the best result..."):
            
            # 3. Call the Hybrid Logic
            try:
                final_response = hybrid_agent_execute(prompt)
                
            except Exception as e:
                final_response = f"A critical error occurred in the Agent pipeline: {e}"
            
            # 4. Display the final response
            st.markdown(final_response) 

    # 5. Save to history
    st.session_state.messages.append({"role": "assistant", "content": final_response})