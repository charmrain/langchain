import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import pickle
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub

import logging
from datetime import datetime

# Set up logging
log_folder = "log_doc"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

log_file = os.path.join(log_folder, "log.txt")
# logging.basicConfig(filename=log_file, level=logging.DEBUG)

# Initialize the logging only once per session
if 'logging_initialized' not in st.session_state:
    logging.basicConfig(filename=log_file, level=logging.INFO)
    st.session_state.logging_initialized = True
    logging.info(f"{datetime.now()} - Application started")


def log(message):
    logging.info(f"{datetime.now()} - {message}")


# Sidebar contents
with st.sidebar:
    st.title('Chatbot of Performix')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - [Performix](https://www.performixbiz.com/) company homepage
 
    ''')
    add_vertical_space(5)
    st.write('test version 1.2c created by Raymond')
    st.write('STATUS: Log in as flame, then systemctl --user status chatai')
    st.write('LOG: journalctl --user -u chatai | less')
    


def main():
    try:
        st.header("Customized Chatbot with PDF files")

        # Log user login time
        log("User logged in")


        load_dotenv()
        

        # upload the pdf files:
        pdffiles = st.file_uploader("Please upload your PDF files:", type='pdf', accept_multiple_files=True)
        # st.write(pdf.name)

        # Initialize session state
        if 'process_button_clicked' not in st.session_state:
            st.session_state.process_button_clicked = False

        # Create a button
        process_button = st.button("Process")

        if pdffiles is not None and process_button:
            st.session_state.process_button_clicked = True

        if st.session_state.process_button_clicked:
            
            # Log PDF upload time
            log("PDF uploaded")


            text =""
            pdfname = []
            for pdf in pdffiles:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
                pdfname.append(pdf.name[:-4])

            st.write("text length: ", len(text))

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )

            
            chunks = text_splitter.split_text(text=text)

            # below is for milvus
            # chunks = text_splitter.create_documents(text)

            # st.write(chunks)
            st.write("number of chunks: ", len(chunks))

            store_name = str(pdfname)
            st.write(store_name)


            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                st.write("embedding loaded from local")
            else:
                embeddings = OpenAIEmbeddings()
                # VectorStore = Milvus.from_documents(documents=chunks, embedding= embeddings, connection_args={"host": 'localhost', "port": "8501"})
                VectorStore = FAISS.from_texts(chunks, embedding= embeddings)

                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
                st.write("embedding is computed from server")

            # accept user's query
            query = st.text_input("Ask questions about the PDF files:")
            # st.write(query)

            if 'prompts' not in st.session_state:
                st.session_state.prompts = []
            if 'responses' not in st.session_state:
                st.session_state.responses = []

            if query:
                docs = VectorStore.similarity_search(query=query,  k=10, fetch_k=20)
            

                llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)


                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    prefix = "according the documents, "
                    userinputs = prefix + query

                    response = chain.run(input_documents=docs, question=userinputs)
                    
                    

                    # meta data test
                    # test more argument in chain.run
                    
                    print(cb)
                    log(cb)


                    st.session_state.prompts.append(query)
                    st.session_state.responses.append(response)
                # st.write(response)
            
            if st.session_state.responses:
                for i in range(len(st.session_state.responses)-1, -1, -1):
                    
                    message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', avatar_style="adventurer",seed=77)
                    message(st.session_state.responses[i], key=str(i), avatar_style="funEmoji", seed='Aneka')
    
    except Exception as e:
        # log the Exception
        logging.exception(f"An error has occurred: {str(e)}")



if __name__=='__main__':
    # Log application start time
    # log("Application started")
    main()


