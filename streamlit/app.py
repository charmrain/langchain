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
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from streamlit_chat import message


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
    st.write('test version 1.1 created by Raymond')
    


def main():
    st.header("Customized Chatbot with PDF files")

    load_dotenv()

    # upload the pdf files:
    pdf = st.file_uploader("Please upload your PDF files:", type='pdf')
    # st.write(pdf.name)

    # check if pdf is exist
    if pdf is not None:

        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader.outline)
        text =""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        
        chunks = text_splitter.split_text(text=text)

        st.write(chunks)

        # embeddings
        # embeddings = OpenAIEmbeddings()

        # VectorStore = FAISS.from_texts(chunks, embedding= embeddings)
        store_name = pdf.name[:-4]
        # st.write(store_name)


        # save the vectorStore as a pickle file
        # check if the pickle file existed
        # if created before, no need to create again

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write("embedding loaded from local")
        else:
            embeddings = OpenAIEmbeddings()
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
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
                st.session_state.prompts.append(query)
                st.session_state.responses.append(response)
            # st.write(response)
          
        if st.session_state.prompts:
            for i in range(len(st.session_state.responses)-1, -1, -1):
                
                message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', seed=77)
                message(st.session_state.responses[i], key=str(i), seed='Milo')


        # resp_text = []
        # resp_text.append(response)

        # st.write("previously asked question:")
        # st.write(resp_text)

        # if 'prompts' not in st.session_state:
        #     st.session_state.prompts = []
        # if 'responses' not in st.session_state:
        #     st.session_state.responses = []
        # def send_click():
        #     if st.session_state.user != '':
        #         prompt = st.session_state.user
        #         if prompt:
        #             docs = knowledge_base.similarity_search(prompt)
        #         llm = OpenAI()
        #         chain = load_qa_chain(llm, chain_type="stuff")
        #         with get_openai_callback() as cb:
        #             response = chain. run(input_documents=docs, question=prompt)
        #         st.session_state.prompts.append(prompt)
        #         st.session_state.responses.append(response)

        




if __name__=='__main__':
    main()


