import streamlit as st 
import os
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI , OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import ChatPromptTemplate 

load_dotenv()

VECTOR_DB_PATH = 'Course_faiss_index'



st.set_page_config(
    page_title= "Academics Support Chatbot",
    page_icon="💼", 
    layout= 'centered'
    )

st.title("Academics Support Bot")

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small', openai_api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_BASE_URL"))
    vectorstore = FAISS.load_local(VECTOR_DB_PATH , embeddings , allow_dangerous_deserialization = True)
    return vectorstore



@st.cache_resource
def load_llm():
    llm = ChatOpenAI(model = 'gpt-4o-mini',temperature =0, openai_api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_BASE_URL"))
    return llm

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = load_llm()

prompt = ChatPromptTemplate.from_template("""
You are a supportive academics assistant. 
Answer the student question using ONLY the course documents. 
If answer is not there in the document then say -
"I don’t have enough information in the provided documents."

Be clear . professional and policy aligned .

Course Content:
{context}

Student Question "
{question}

""")

if "messages" not in st.session_state :
    st.session_state.messages =[]

for msg in st.session_state.messages :
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

user_question = st.chat_input("Ask a question related to course...")


if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)

    docs = retriever.invoke(user_question)
    context = "\n\n".join(doc.page_content for doc in docs)
    response = llm.invoke(prompt.format_messages(context=context, question=user_question))

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.messages.append({"role": "assistant", "content": response.content})