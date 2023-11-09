from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.session import Session
from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

from dotenv import load_dotenv

load_dotenv()

# Define your content function here if it's not already defined
def get_content_func(data, **_):
    return data.get("prompt").split("Question")[-1]

def setup_session():
    session = Session(name="sqlite-example")
    return session

def setup_database(session):
    db = SQLDatabase.from_uri("sqlite:///./Chinook.db")
    return db

def setup_lang_model(session):
    llm = LangChainLLMs(llm=OpenAI(temperature=0), session=session)
    return llm

def setup_database_chain(llm, database):
    db_chain = SQLDatabaseChain(llm=llm, database=database, verbose=True)
    return db_chain

def setup_cache():
    onnx = Onnx()
    cache_base = CacheBase('sqlite')
    vector_base = VectorBase('milvus', host='127.0.0.1', port='19530', dimension=onnx.dimension)
    data_manager = get_data_manager(cache_base, vector_base)
    cache.init(
        pre_embedding_func=get_content_func,
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
    )
    cache.set_openai_key()

def run_query(db_chain, query):
    result = db_chain.run(query)
    return result

def main():
    session = setup_session()
    database = setup_database(session)
    lang_model = setup_lang_model(session)
    database_chain = setup_database_chain(lang_model, database)
    
    # Set up the cache
    setup_cache()
    
    query = "How many employees are there?"
    result = run_query(database_chain, query)
    
    # Use the result as needed
    
if __name__ == "__main__":
    main()

