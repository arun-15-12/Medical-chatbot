from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


def load_pdf_files(data):
    loader=DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents



def filter_to_minimal_docs(docs:List[Document])-> List[Document]:
    """ given a list of documents return a new list of documents objects containing only 'sourece'
     in metadata and the original page content  """
    minimal_docs : List[Document]=[]
    for doc in docs:
        src= doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



### chucnking operation
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk



##converting to embeddings 
from langchain_huggingface import HuggingFaceEmbeddings
def download_embeddings():
    """ Downlaod and return the huggingface embeddings model """
    model_name="BAAI/bge-small-en-v1.5"

    embeddings=HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device":"cpu"}
    )
    return embeddings

embeddings=download_embeddings()




# --- ADD THIS TO THE BOTTOM OF helper.py ---

# 1. Load data
extracted_data = load_pdf_files(data="data/")

# 2. Filter and split
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)

# 3. Initialize embeddings
embeddings = download_embeddings()



