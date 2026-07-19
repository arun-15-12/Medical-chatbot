from src.helper import texts_chunk, embeddings
from dotenv import load_dotenv
import os 
load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"]=MISTRAL_API_KEY

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"]=MISTRAL_API_KEY

from pinecone import Pinecone
pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)

# standard syntax
from  pinecone import ServerlessSpec
index_name= "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384, # dimension of the embedding 
        metric="cosine", # cosine similarity 
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore
docsearch =PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name
)








