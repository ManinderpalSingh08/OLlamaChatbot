import  streamlit as st
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


llm = Ollama(model="llama3",temperature=0.7)  
memory = ConversationBufferMemory()

template = """
The following is a conversation between a human and an AI assistant.
The AI assistant is helpful, polite, and informative.

Human: {user_input}
AI:
"""
prompt = PromptTemplate(template=template, input_variables=["user_input"])

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

user_input = st.text_input("Type your message:")


if st.button("Send"):

    response = chain.run({"user_input": user_input})
    st.text_area("Conversation", memory.buffer, height=300) 