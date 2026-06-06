import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.prompts import PromptTemplate
load_dotenv()

app = FastAPI(title="Legal Transition Engine API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Connect to the exact same embedding model configuration
# Notice the slightly different import name:
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

print("Initializing BGE embedding model...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    # This single line fixes the entire hallucination issue:
    query_instruction="Represent this sentence for searching relevant passages: " 
)

# Initialize your high-performance LLM via Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Connect to the disk-persisted vector database
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# Grabbing the top 6 most relevant chunks instead of 3
# Upgraded from standard similarity to Maximal Marginal Relevance (MMR)
# 1. Your existing MMR Base Retriever
# 1. Your existing MMR Base Retriever (Tuned for Multi-Query)
# 1. Base Retriever: Strict Similarity, Top 1 per query
# 1. Base Retriever: Strict Similarity with a Bouncer (Score Threshold)
base_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 6, 
        "score_threshold": 0.5
    }
)
    # 2. The Custom Legal Translation Prompt
query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an expert Indian legal assistant.
        Your task is to take a user's conversational legal question and rewrite it into 3 distinct, highly technical legal queries using formal Bharatiya Nyaya Sanhita (BNS) terminology.
        
        CRITICAL RULE: You MUST preserve the specific core facts of the scenario in EVERY query (e.g., if it involves a vehicle, mention 'driving' or 'vehicle'. If it involves a weapon, mention 'deadly weapon'). Do not generate overly broad or generic queries.
        
        Original question: {question}
        
        Generate 3 technical legal queries separated by newlines:"""
    )

    # 3. The Advanced Multi-Query Retriever
advanced_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm, # This uses your existing LLM to rewrite the prompt!
        prompt=query_prompt
    )

class ChatRequest(BaseModel):
    query: str

# Engineering strict system prompts protects against hallucinations
template = """You are an expert Indian criminal defense attorney and legal scholar. 
Analyze the user's factual scenario strictly using the provided legal context.
Note: IPC stands for the Indian Penal Code. BNS stands for the Bhartiya Nyaya Sanhita.
If the user asks about an old IPC section, clearly provide its new BNS equivalent based on the context.
Always explicitly cite the Code name (IPC/BNS) and Section numbers.

Context:
{context}

User Query: {question}

Legal Analysis:"""
prompt = ChatPromptTemplate.from_template(template)
def format_docs(docs):
    # This explicitly feeds the mapping (e.g. "[IPC 378 -> BNS 303 - Theft]") directly into the LLM's brain
    return "\n\n".join(
        f"[IPC Sec {doc.metadata.get('ipc_section', 'N/A')} -> BNS Sec {doc.metadata.get('section', 'N/A')} - {doc.metadata.get('title', '')}]: {doc.page_content}" 
        for doc in docs
    )

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Retrieve context matches from Chroma
        retrieved_docs = advanced_retriever.invoke(request.query)
        context_str = format_docs(retrieved_docs)
        
        # Chain execution via LCEL
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "context": context_str,
            "question": request.query
        })
        
        return {
            "answer": response,
            "sources": [
                {**doc.metadata, "text": doc.page_content} for doc in retrieved_docs
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))