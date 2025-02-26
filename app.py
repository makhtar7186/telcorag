from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
#from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity

from langchain.schema import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq.chat_models import ChatGroq
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Charger les variables d'environnement
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialisation des composants
persist_directory = "./chroma_storage"

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
naive_chunk_vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=persist_directory
)
naive_chunk_retriever = naive_chunk_vectorstore.as_retriever(search_kwargs={"k": 2})

chat_model = ChatGroq(
    temperature=0.3,
    model_name="mixtral-8x7b-32768",
    api_key=os.getenv("GROQ_API_KEY"),
)

rag_template = """\
Use the following context to answer the user's query. If you cannot answer, please respond with 'I don't know'.

User's Query:
{question}

Context:
{context}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

def context_retriv(query):
    context = str(naive_chunk_retriever.invoke(query))
    return context
naive_rag_chain = (
    {"context": context_retriv, "question": RunnablePassthrough()}
    | rag_prompt
    | chat_model
    | StrOutputParser()
)

# Charger le modèle d'encodage
model = SentenceTransformer("all-MiniLM-L6-v2")

# Fonction pour calculer la similarité cosinus
def calculate_similarity(question, document):
    # Encoder la question et le document
    question_embedding = model.encode(question, convert_to_tensor=True)
    document_embedding = model.encode(document, convert_to_tensor=True)
    
    # Déplacer les tenseurs sur le CPU et convertir en numpy
    question_embedding = question_embedding.cpu().detach().numpy()
    document_embedding = document_embedding.cpu().detach().numpy()
    
    # Calculer la similarité cosinus
    similarity = cosine_similarity(
        [question_embedding],
        [document_embedding]
    )
    return similarity[0][0]  # Retourner le score de similarité

###########################################
query_gen_str = """\
You are a helpful assistant that generates multiple search queries based on a \
single input query. Generate {num_queries} search queries, one on each line, \
related to the following input query:
Query: {query}
Queries:
"""

query_gen_prompt = ChatPromptTemplate.from_template(query_gen_str)


# Fonction pour générer des requêtes
def generate_queries(query: str, llm, num_queries: int = 4):
    # Générer le prompt en passant les variables nécessaires
    prompt = query_gen_prompt.format(num_queries=num_queries, query=query)
    
    # Appeler le modèle pour prédire
    response = llm.predict(prompt)
    
    # Assumer que chaque requête est sur une nouvelle ligne
    queries = response.split("\n")
    queries_str = "\n".join(queries)
     
    return queries

def get_cont(query):
    Mquery = generate_queries(query, chat_model)
    contg= []
    for query in Mquery:
        contex= naive_chunk_retriever.invoke(query)
        contg.append(contex)
    context = "\n".join([str(c) for c in contg])
    return context
  
multi_naive_rag_chain = (
    {"context" : get_cont, "question" : RunnablePassthrough()}
    | rag_prompt
    | chat_model
    | StrOutputParser()
)
##########################################
# Définir FastAPI
app = FastAPI()

# Modèle de requête
class QueryRequest(BaseModel):
    question: str

# Endpoint principal
@app.post("/query/")
async def query_pipeline(request: QueryRequest):
    try:
        result = multi_naive_rag_chain.invoke(request.question)
        context = get_cont(request.question)
        similarite = calculate_similarity(result,request.question)
        similarity = float(similarite)
        return {"question": request.question, "response": result,"similarity": similarity}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {e}")
#uvicorn app:app --reload --host 0.0.0.0 --port 8000