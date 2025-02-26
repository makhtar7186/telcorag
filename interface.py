import streamlit as st
import requests

# Configurer l'URL de l'API
API_URL = "http://127.0.0.1:8000/query/"

# Configurer la page Streamlit
st.set_page_config(page_title="RAG Pipeline - Question Answering", page_icon="🤖", layout="wide")

# Initialiser l'historique des questions/réponses dans la session
if "history" not in st.session_state:
    st.session_state["history"] = []

# Fonction pour appeler l'API
def query_api(question):
    try:
        response = requests.post(API_URL, json={"question": question})
        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", "Aucune réponse trouvée.")
            similarity = data.get("similarity", "Score de similarité non disponible.")
            return answer, similarity
        else:
            return f"Erreur API : {response.json().get('detail', 'Problème inconnu')}", None
    except requests.exceptions.RequestException as e:
        return f"Erreur de connexion à l'API : {e}", None

# Section principale
st.title("🤖 3GPP/UIT RAG LLM")
st.markdown("Posez une question et obtenez des réponses avec des informations contextuelles pertinentes.")

# Disposition en colonnes
col1, col2 = st.columns([2, 1])

# Section pour poser une question
with col1:
    st.header("📋 Posez votre question")
    question = st.text_input("Posez votre question :", placeholder="Exemple : Quelle est la vitesse de la lumière ?")
    if st.button("Envoyer"):
        if question.strip():
            with st.spinner("Recherche en cours..."):
                answer, similarity = query_api(question)
                if similarity is not None:
                    st.success("Réponse obtenue avec succès.")
                    st.markdown(f"**Réponse :** {answer}")
                    st.markdown(f"**Score de similarité :** {similarity:.2f}")
                else:
                    st.warning(f"**Erreur :** {answer}")
                # Ajouter la question et la réponse à l'historique
                st.session_state["history"].append({"question": question, "answer": answer})
        else:
            st.warning("Veuillez entrer une question.")

# Historique des interactions
with col2:
    st.header("📜 Historique")
    if st.session_state["history"]:
        for i, interaction in enumerate(st.session_state["history"][::-1]):  # Affiche les plus récentes en haut
            with st.expander(f"Question #{len(st.session_state['history']) - i}: {interaction['question']}"):
                st.markdown(f"**Réponse :** {interaction['answer']}")
    else:
        st.info("L'historique est vide. Posez une question pour commencer !")
