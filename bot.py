import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Télécharger les ressources nécessaires de NLTK (si vous ne l'avez pas déjà fait)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#=========================================================================================================================================================

def load_and_preprocess_text(file_path):
    """ Charger le fichier texte et le prétraiter """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().replace('\n', ' ')
    sentences = sent_tokenize(data)
    preprocessed_sentences = [preprocess(sentence) for sentence in sentences]
    return preprocessed_sentences

def preprocess(sentence):
    """ Fonction de prétraitement pour chaque phrase """
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

def get_most_relevant_sentence(query, corpus):
    """ Fonction pour trouver la phrase la plus pertinente """
    query_processed = preprocess(query)
    max_similarity = 0
    most_relevant_sentence = ""
    
    for sentence in corpus:
        similarity = len(set(query_processed).intersection(set(sentence))) / float(len(set(query_processed).union(set(sentence))))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    
    return most_relevant_sentence

def main():
    """ Créer l'application Streamlit """
    st.title("Chatbot")
    st.write("Bonjour! Je suis un chatbot. Demandez-moi quoi que ce soit sur le sujet dans le fichier texte.")
    
    # Charger et prétraiter le fichier texte
    file_path = 'ww2.txt'  # Remplacez par le chemin correct de votre fichier
    preprocessed_sentences = load_and_preprocess_text(file_path)
    
    # Obtenir la question de l'utilisateur
    question = st.text_input("Vous:")
    
    # Créer un bouton pour soumettre la question
    if st.button("Soumettre"):
        if question:
            response = get_most_relevant_sentence(question, preprocessed_sentences)
            st.write("Chatbot: " + response)
        else:
            st.write("Veuillez entrer une question.")

if __name__ == "__main__":
    main()
