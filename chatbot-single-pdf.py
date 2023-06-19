import streamlit as st 
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

with st.sidebar:
    st.markdown("""
                ## About
                This is a chatbot application where you can chat with a single PDF document.
                The Application is built with
                + Langchain
                + OpenAI
                + Streamlit (for UI)
                
                """)
    
def split_docs(text):
    text_splitter = RecursiveCharacterTextSplitter(['\n\n', '\n', ' ', ''], chunk_size=500, chunk_overlap=50, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def create_index(chunks):
    embedding = OpenAIEmbeddings()
    db = Chroma.from_texts(chunks, embedding)
    return db
    

def main():
    st.header('Chat with Single PDF Document')
    col1, col2 = st.columns([0.3, 0.7])
    pdf = st.sidebar.file_uploader('Upload your PDF file', type='pdf')
    
    st.session_state.texts = ''
    st.session_state.request = ''
    
    if pdf is not None:
        pdf_pages = PdfReader(pdf)
        for page in pdf_pages.pages:
            st.session_state.texts += page.extract_text()
    
        query = st.text_input('Ask a question about the PDF')
        if query:
            st.session_state.request = query
        
    
        st.subheader('Document Content')
        docs = st.session_state.texts
        text_chunks = split_docs(docs)
        index = create_index(text_chunks)
        query = st.session_state.request
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=index.as_retriever(), chain_type='stuff')
        response = qa.run(query)
        st.write(response)
        
    
if __name__=='__main__':
    main()