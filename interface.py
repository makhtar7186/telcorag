import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

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

# Charger le modèle d'encodage
model = SentenceTransformer("all-MiniLM-L6-v2")

# Fonction pour calculer la similarité cosinus
def calculate_similarity(question, document):
    question_embedding = model.encode(question, convert_to_tensor=True)
    document_embedding = model.encode(document, convert_to_tensor=True)
    question_embedding = question_embedding.cpu().detach().numpy()
    document_embedding = document_embedding.cpu().detach().numpy()
    similarity = cosine_similarity([question_embedding], [document_embedding])
    return similarity[0][0]

# Fonction pour générer des requêtes
def generate_queries(query: str, llm, num_queries: int = 4):
    query_gen_str = """\
    You are a helpful assistant that generates multiple search queries based on a \
    single input query. Generate {num_queries} search queries, one on each line, \
    related to the following input query:
    Query: {query}
    Queries:
    """
    query_gen_prompt = ChatPromptTemplate.from_template(query_gen_str)
    prompt = query_gen_prompt.format(num_queries=num_queries, query=query)
    response = llm.predict(prompt)
    queries = response.split("\n")
    return queries

# Fonction pour récupérer le contexte
def get_cont(query):
    Mquery = generate_queries(query, chat_model)
    contg = []
    for q in Mquery:
        contex = naive_chunk_retriever.invoke(q)
        contg.append(contex)
    context = "\n".join([str(c) for c in contg])
    return context

# Chaîne RAG
multi_naive_rag_chain = (
    {"context": get_cont, "question": RunnablePassthrough()}
    | rag_prompt
    | chat_model
    | StrOutputParser()
)

# Interface Streamlit
st.set_page_config(page_title="RAG Pipeline - Question Answering", page_icon="🤖", layout="wide")

if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("🤖 3GPP/UIT RAG LLM")
st.markdown("Posez une question et obtenez des réponses avec des informations contextuelles pertinentes.")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Posez votre question")
    question = st.text_input("Posez votre question :", placeholder="Exemple : Quelle est la vitesse de la lumière ?")
    if st.button("Envoyer"):
        if question.strip():
            with st.spinner("Recherche en cours..."):
                result = multi_naive_rag_chain.invoke(question)
                context = get_cont(question)
                similarity = calculate_similarity(result, question)
                st.success("Réponse obtenue avec succès.")
                st.markdown(f"**Réponse :** {result}")
                st.markdown(f"**Score de similarité :** {similarity:.2f}")
                st.session_state["history"].append({"question": question, "answer": result})
        else:
            st.warning("Veuillez entrer une question.")

with col2:
    st.header("📜 Historique")
    if st.session_state["history"]:
        for i, interaction in enumerate(st.session_state["history"][::-1]):
            with st.expander(f"Question #{len(st.session_state['history']) - i}: {interaction['question']}"):
                st.markdown(f"**Réponse :** {interaction['answer']}")
    else:
        st.info("L'historique est vide. Posez une question pour commencer !")
