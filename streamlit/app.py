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


# from langchain.evaluation.qa import QAGenerateChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.llms import HuggingFaceHub

from langchain.vectorstores import Milvus

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
    st.write('test version 1.2 created by Raymond')
    


def main():
    st.header("Customized Chatbot with PDF files")


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
    # if st.button("Process"):
        # with st.spinner("Processing"):
            # st.write(pdf_reader.outline)
    if st.session_state.process_button_clicked:
        text =""
        pdfname = []
        for pdf in pdffiles:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
            pdfname.append(pdf.name[:-4])
            
        

        # st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
            # chunk_size=500,
            # chunk_overlap=0,
        
        chunks = text_splitter.split_text(text=text)

        # below is for milvus
        # chunks = text_splitter.create_documents(text)

        st.write(chunks)

        # embeddings
        # embeddings = OpenAIEmbeddings()

        # VectorStore = FAISS.from_texts(chunks, embedding= embeddings)
        store_name = str(pdfname)
        st.write(store_name)


        # save the vectorStore as a pickle file
        # check if the pickle file existed
        # if created before, no need to create again

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
            # Here is an example of how to set fetch_k parameter when calling similarity_search. 
            # Usually you would want the fetch_k parameter >> k parameter. 
            # This is because the fetch_k parameter is the number of documents that will be fetched before filtering. 
            # If you set fetch_k to a low number, you might not get enough documents to filter from.
            # https://python.langchain.com/docs/integrations/vectorstores/faiss
            # https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.faiss.FAISS.html#langchain.vectorstores.faiss.FAISS.similarity_search
            
            # try different k value
            # fetch_k=4, k=1
            # check the filter arg

            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
            
            # hugging face
            # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 64})
            # https://python.langchain.com/docs/integrations/llms/huggingface_hub
            
            # gpt-4
            # llm = OpenAI(model_name='davinci-002')
            # llm = OpenAI(model_name='gpt-4-0314')
            # https://platform.openai.com/docs/models/gpt-4


            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                # meta data test
                # test more argument in chain.run
                
                print(cb)
                st.session_state.prompts.append(query)
                st.session_state.responses.append(response)
            # st.write(response)
        
        if st.session_state.responses:
            for i in range(len(st.session_state.responses)-1, -1, -1):
                
                message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', avatar_style="adventurer",seed=77)
                message(st.session_state.responses[i], key=str(i), avatar_style="funEmoji", seed='Aneka')



if __name__=='__main__':
    main()


