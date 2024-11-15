# LangChain Chatbot(Llama3) and LLM Optimization Guide

This repository contains two Streamlit applications:

1. *Chatbot with LangChain and Ollama:* A simple chatbot built using LangChain and Ollama, allowing you to interact with a Llama 3 8b model.

2. *LLM Optimization Guide:* A Streamlit documentation page providing guidance on how to optimize your LLMs using LangChain techniques.

## Getting Started

1. *Clone the Repository:*
   
    `git clone https://github.com/AliBeiramiii/Chatbot-Ollama3.git`
   
2. *Install Dependencies:*

 `pip install -r requirements.txt`

3. *Run Ollama:*
   * `curl -fsSL https://ollama.com/install.sh | sh`
   * `ollama pull llama3`

4. *Run the Chatbot:*

   `streamlit run src/chatbot.py`

5. *Run the Optimization Guide:*

   `streamlit run src/main.py`

## Features

*Chatbot:*

• *Interactive Chat:*  Allows you to chat with a Llama 3 model.

• *Conversation History:*  Stores the conversation history for context.(In Progress)

*LLM Optimization Guide:*

• *Memory and Context:*  Covers techniques for managing conversation history and external knowledge.

• *Parameter Tuning:*  Explains how to adjust parameters like temperature, top_k, and top_p.


## Contributing

Contributions are welcome!
