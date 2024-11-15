import streamlit as st


def page1():
    st.title("Fine-tune")
    st.write("""LangChain focuses on *using* and *combining* pre-trained LLMs. It provides tools for:
             it doesn't directly handle the fine-tuning process itself.
             """)
    
def page2():
    st.title("Memory")
    st.write("""   * **Conversation History:**  Use a `ConversationBufferMemory` to store the conversation history and pass it to the LLM, allowing it to maintain context.
   * **External Knowledge:**  Integrate external knowledge sources (like databases or APIs) into your chains to provide the LLM with additional information.""")

    code1 = """from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain


llm = Ollama(model="llama3", temperature=0.0)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)"""

    st.write('- **Save previous chats in buffer:**')
    st.code(code1, language="python")
    
    code2= """from langchain.memory import ConversationBufferWindowMemory


# k determines how many previous conversations to remember
memory = ConversationBufferWindowMemory(k=1)"""

    st.write('- **Short-term memory:**')
    st.code(code2, language="python")
    
    code3 = """from langchain.memory import ConversationTokenBufferMemory


# Different llms have different tokenization process
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)"""

    st.write('- **Certain number of toknes in the memory**')
    st.code(code3, language="python")
    
    code4 = """from langchain.memory import ConversationSummaryBufferMemory


# Different llms have different tokenization process
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=50) """

    st.write('- **Summerize the conversation in limited tokens:**')
    st.code(code4, language="python")
    
    code5 = """memory.save_context({'input':'To remember'},
                    {'output':'Something'})"""
                    
    st.write('- **External Knowledge :**')
    st.code(code5, language="python")
    
def page3():
    
    st.title("Temperature")
    st.write("""* **Temperature:**  Adjust the `temperature` parameter of the LLM to control the randomness of its responses. Higher temperatures lead to more creative and unpredictable outputs.
   * **Top_k and Top_p:**  Use `top_k` and `top_p` to filter the model's output and reduce the likelihood of generating nonsensical text.
   How Temperature Affects Output:

- **Low Temperature** (Close to 0):
   * More Predictable:  The model will tend to choose the most likely words, resulting in more predictable and conservative outputs. This is good for tasks where accuracy and consistency are important.
   * Example:  A chatbot responding with straightforward answers.

- **High Temperature** (Greater than 1):
   * More Creative and Risky:  The model will be more likely to choose less probable words, leading to more creative and surprising outputs. This can be helpful for tasks like creative writing or generating diverse ideas.
   * Example:  A chatbot generating stories or poems.
- A lower top_k or top_p might be better for providing more focused and coherent responses.
- A higher top_k or top_p might be more suitable for generating more diverse and surprising outputs.""")

    code1  = """low_temp_llm = Ollama(model="llama3", temperature=0.2)"""

    code2 = """high_temp_llm = Ollama(model="llama3", temperature=0.8)"""
    
    st.code(code1, language='python')
    st.code(code2, language='python')
    
def page4():
    
    st.title("Evaluating")
    st.write("""1. Define Evaluation Metrics:

   * Choose Appropriate Metrics: The best metrics depend on your specific task and the type of LLM you're using:
      * Accuracy:  Percentage of correct predictions (for tasks like classification, question answering).
      * Perplexity:  Measures how well the model predicts the next word in a sequence. Lower perplexity indicates a better fit to the data.
      * BLEU Score:  Evaluates the quality of generated text by comparing it to reference translations or human-written text.

2. Test Dataset:

   * Representative Data:  Your test dataset should be distinct from the training data and represent the types of inputs your LLM will encounter in real-world use.
   * Example:
      * For a chatbot, use a set of conversations with diverse topics and questions.
      * For a text summarization model, use a set of documents and their corresponding summaries.

3. Use LangChain's Evaluation Tools:

   * LangChain's evaluate module: LangChain offers tools for evaluating different types of LLMs:
      * `evaluate.llm_evaluation`: Provides functions for evaluating LLMs on various tasks, including:
         * `evaluate_llm_on_evals`:  Evaluates an LLM using a pre-defined set of evaluation tasks.
         * `evaluate_llm_on_dataset`: Evaluates an LLM using a custom dataset.
         * `evaluate_llm_on_prompt`: Evaluates an LLM on a single prompt.
      * `evaluate.chains`: Provides functions for evaluating LangChain chains, which can be useful for tasks like question answering, summarization, and more.""")
        
    

st.sidebar.title("Optimizing llm:")

menu_items = ["Fine-tune", "Memory", "Temperature","Evaluating"]

for item in menu_items:
    if st.sidebar.button(item):
        if item == "Fine-tune":
            page1()
        elif item == "Memory":
            page2()
        elif item == "Temperature":
            page3()
        elif item == "Evaluating":
            page4()
