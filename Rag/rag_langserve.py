import os
from dotenv import load_dotenv ,find_dotenv

GROQ_API_KEY="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"


NEO4J_URI='neo4j+s://2edf4c18.databases.neo4j.io'
NEO4J_USERNAME='neo4j'
NEO4J_PASSWORD='sDmd3TTYDjmmn3f-c0I_mOR0pdIgc0tQQxghJrLAH50'


# _ = load_dotenv(find_dotenv())
# GROQ_API_KEY=os.getenv("GROQ_API_KEY")
# from langchain_groq import ChatGroq


# NEO4J_URI=os.getenv('NEO4J_URI')
# NEO4J_USERNAME=os.getenv('NEO4J_USERNAME')
# NEO4J_PASSWORD=os.getenv('NEO4J_PASSWORD')

# all the libraries
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

## menmory libraries
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

## libraries for front end
from fastapi import FastAPI
from langserve import add_routes
import uvicorn


## load the web page
loader=WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content","post_title","post-header")
            
            
        )
    ),
)
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
#documnet into chunks or splits
splits=text_splitter.split_documents(docs)


## create the embeddings
model="BAAI/bge-large-en"
embeddings = HuggingFaceEmbeddings(model_name=model)

## create the vector store
#create a Neo4j vector store
neo4j_db=Neo4jVector.from_documents(
    splits,
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database='neo4j',
    index_name='lcel_index',
    node_label='lcel_node',
    text_node_property="text",
    embedding_node_property="embeddings",
      
)


##load the model
##load llm
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY,
    
)

## retriver

retriever = neo4j_db.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


## fast api front end
app=FastAPI(
    title="rag_langserve",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    rag_chain,
    path="/rag_chain",
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

