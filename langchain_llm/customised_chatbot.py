import os
import sys
import openai

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader



def data_ingress():
    """ this compenent aims to incorprate multiple pdf files
    those files can provide the basic information to chatbots,
    when the chatbox answer any questions, it will use the information 
    that given by the pdf files

    input: pdf.files
    output: an object that langchain can digest
    """

    loader = PyPDFLoader("*.pdf")
    pages = loader.load_and_split()

    pass


def langchain_chatbot():
    """ this compenent is to implement Langchain to train a chatbot with the pdf files
    the chatbot is to digest the knowledge shared by the data_ingress function

    input: data_ingress()
    output: langchain chatbox
    """
    pass



def evaluation():
    """ this compenent aims to find out the quality of the chatbox
    the question that we know the answer will pitch to the chatbot,
    see how many questions the chatbot can answer correctly

    input: questions for the chatbot
    output: the accuracy rate and error answer
    """
    pass


if __name__ == '__main__':
    pass