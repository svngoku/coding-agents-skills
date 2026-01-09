# Retrieval & RAG Patterns

Build retrieval-augmented generation systems.

## Basic RAG

```python
from langchain.vectorstores import InMemoryVectorStore
from langchain.embeddings import init_embeddings
from langchain.agents import create_agent
from langchain.tools import tool

# Setup vector store
embeddings = init_embeddings("openai:text-embedding-3-small")
vectorstore = InMemoryVectorStore(embeddings)

# Add documents
vectorstore.add_texts([
    "Company policy: Remote work allowed 2 days/week",
    "Benefits include health insurance and 401k match"
])

@tool
def search_docs(query: str) -> str:
    """Search company documents."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join(doc.page_content for doc in docs)

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[search_docs],
    system_prompt="Answer using retrieved documents."
)
```

## Document Loading

```python
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader
)

# PDF
docs = PyPDFLoader("document.pdf").load()

# Text
docs = TextLoader("file.txt").load()

# CSV
docs = CSVLoader("data.csv").load()

# Web
docs = WebBaseLoader("https://example.com").load()
```

## Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(docs)
```

## Vector Stores

```python
from langchain.vectorstores import (
    InMemoryVectorStore,
    Chroma,
    FAISS,
    Pinecone
)

# In-memory (dev)
vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)

# Chroma (local persistence)
vectorstore = Chroma.from_documents(
    docs, embeddings, 
    persist_directory="./chroma_db"
)

# FAISS (fast similarity search)
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")

# Pinecone (production scale)
from pinecone import Pinecone
pc = Pinecone(api_key="...")
vectorstore = Pinecone.from_documents(
    docs, embeddings,
    index_name="my-index"
)
```

## Retrievers

```python
# Basic retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# With score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)

# MMR (diversity)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

## Embedding Models

```python
from langchain.embeddings import init_embeddings

# OpenAI
embeddings = init_embeddings("openai:text-embedding-3-small")

# Anthropic (via Voyage)
embeddings = init_embeddings("voyageai:voyage-3")

# Local (Ollama)
embeddings = init_embeddings("ollama:nomic-embed-text")

# Cohere
embeddings = init_embeddings("cohere:embed-english-v3.0")
```

## RAG Agent with Context

```python
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

@tool
def search_knowledge_base(query: str, runtime: ToolRuntime) -> str:
    """Search knowledge base with user context."""
    user_dept = runtime.context.department
    
    # Filter by department
    docs = vectorstore.similarity_search(
        query,
        k=5,
        filter={"department": user_dept}
    )
    return format_docs(docs)

@tool
def search_with_history(query: str, runtime: ToolRuntime) -> str:
    """Search considering conversation history."""
    messages = runtime.state["messages"]
    
    # Expand query with context
    context = extract_context(messages[-5:])
    expanded = f"{context} {query}"
    
    return vectorstore.similarity_search(expanded, k=3)
```

## Hybrid Search

Combine keyword + semantic search:

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# Keyword retriever
bm25 = BM25Retriever.from_documents(docs)
bm25.k = 5

# Semantic retriever
semantic = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine
hybrid = EnsembleRetriever(
    retrievers=[bm25, semantic],
    weights=[0.4, 0.6]
)
```

## Self-Query Retriever

Natural language to structured filters:

```python
from langchain.retrievers.self_query import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="year", type="integer", description="Publication year"),
    AttributeInfo(name="author", type="string", description="Author name"),
    AttributeInfo(name="category", type="string", description="Document category")
]

retriever = SelfQueryRetriever.from_llm(
    llm=model,
    vectorstore=vectorstore,
    document_contents="Technical documentation",
    metadata_field_info=metadata_field_info
)

# Query: "papers by Smith after 2020" 
# Auto-generates: filter={"author": "Smith", "year": {"$gt": 2020}}
```

## Contextual Compression

Reduce retrieved content to relevant parts:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(model)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```