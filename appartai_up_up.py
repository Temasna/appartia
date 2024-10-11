# -*- coding: utf-8 -*- 
"""
Created on Sat Oct  5 13:54:11 2024

@author: sametn
"""

# app.py

import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import json
from PyPDF2 import PdfReader

# Mettre à jour les importations pour correspondre aux dernières versions de langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

def main():
    st.title("AppartAI - Assistant Intelligent")
    st.write("Posez vos questions sur l'appartement, et obtenez des réponses instantanément.")

    # Charger la clé API OpenAI depuis le fichier .env
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        st.error("La clé API OpenAI n'a pas été trouvée. Veuillez vérifier que le fichier .env contient la clé.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Chemin du fichier pour enregistrer l'historique
    history_file = "conversation_history.json"

    # Charger l'historique existant si le fichier existe
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            st.session_state.history = json.load(f)
    else:
        st.session_state.history = []

    # Importer les documents pour chaque appartement séparément
    uploaded_files = st.file_uploader("Charger les fichiers PDF contenant les informations des appartements", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        appartement_docs = []
        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() + "\n"
            appartement_docs.append(pdf_text)

        # Préparer le système avec les documents importés
        embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = [text_splitter.create_documents([text]) for text in appartement_docs]

        # Champ de sélection pour choisir un appartement
        appartement_selectionne = st.selectbox("Choisissez l'appartement qui vous intéresse", [f"Appartement {i+1}" for i in range(len(docs))])
        st.write(f"Vous avez sélectionné : {appartement_selectionne}")

        # Filtrer les documents pour l'appartement sélectionné
        selected_index = int(appartement_selectionne.split()[-1]) - 1
        selected_docs = docs[selected_index]

        vectorstore = FAISS.from_documents(selected_docs, embeddings)
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Champ de saisie pour la question
        query = st.text_input("Votre question :", "")

        if query:
            with st.spinner('Traitement en cours...'):
                try:
                    answer = qa_chain.run(query)
                    # Ajouter la question et la réponse à l'historique
                    new_entry = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "appartement": appartement_selectionne,
                        "question": query,
                        "answer": answer
                    }
                    st.session_state.history.append(new_entry)
                    # Limiter l'historique à 100 entrées (optionnel)
                    st.session_state.history = st.session_state.history[-100:]
                    # Enregistrer l'historique dans le fichier
                    with open(history_file, 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.history, f, ensure_ascii=False, indent=4)
                    # Réinitialiser le champ de saisie
                    #st.experimental_rerun()
                except Exception as e:
                    st.error(f"Une erreur s'est produite : {e}")

        # Afficher l'historique de la conversation (le plus récent en haut)
        if st.session_state.history:
            st.subheader("Historique de la conversation")
            for chat in reversed(st.session_state.history):
                appartement = chat.get('appartement', 'Non spécifié')
                st.markdown(f"*{chat['time']}* - **{appartement}**")
                st.markdown(f"**Vous :** {chat['question']}")
                st.markdown(f"**Assistant :** {chat['answer']}"
)

if __name__ == "__main__":
    main()