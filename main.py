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
import zipfile
import os

if os.path.exists("chroma_db.zip") and not os.path.exists("chroma_db"):
    with zipfile.ZipFile("chroma_db.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Database unzipped successfully!")
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

print("Initializing BGE embedding model...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    query_instruction="Represent this sentence for searching relevant passages: " 
)


groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
   
)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# In main.py
base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4} 
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
        llm=llm, 
        prompt=query_prompt
    )

class ChatRequest(BaseModel):
    query: str

template = """You are an expert Indian criminal defense attorney. 
STRICT INSTRUCTION: You must answer the user's question using ONLY the provided 'Context' below.
If the answer is not contained within the provided context, state that you do not have that specific information in your database.
DO NOT use your internal training data to guess section numbers.
Cite the BNS section clearly as '[BNS Sec X]'.

Context:
{context}

User Query: {question}
Legal Analysis:"""
prompt = ChatPromptTemplate.from_template(template)
def format_docs(docs):
    
    print(f"DEBUG: Retrieved {len(docs)} documents.")
    for d in docs:
        print(f"DEBUG: Found {d.metadata.get('section')}")
        
    return "\n\n".join(
        f"[BNS Sec {doc.metadata.get('section', 'N/A')}]: {doc.page_content}" 
        for doc in docs
    )

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Retrieve candidates without filtering
    docs = vectorstore.similarity_search_with_score(request.query, k=10)
    print("--- SEARCH RESULTS ---")
    for doc, score in docs:
        print(f"Section: {doc.metadata.get('section')} | Score: {score}")
    print("----------------------")
    

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
        
       
        formatted_sources = []
        for doc in retrieved_docs:
            source_data = {**doc.metadata, "text": doc.page_content}
            
            # If the database forgot to label the code type, force it to "BNS"
            if "code_type" not in source_data:
                source_data["code_type"] = "BNS"
                
            # Ensure the section number is treated as text, not a math integer
            if "section" not in source_data:
                source_data["section"] = str(source_data.get("ipc_section", "N/A"))
                
            formatted_sources.append(source_data)

        return {
            "answer": response,
            "sources": formatted_sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
