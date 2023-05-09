import os

# Set the current working directory to the directory where main.py is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from tkinter import *
from tkinter import ttk,messagebox
import webbrowser
import speech_recognition
from pygame import  mixer
#Indexation:
import os
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np


mixer.init()

def search():
    if questionField.get()!='':
        if temp.get()=='google':
            webbrowser.open(f'https://www.google.com/search?q={questionField.get()}')
            print(questionField.get())

        if temp.get()=='duck':
            webbrowser.open(f'https://duckduckgo.com/?q={questionField.get()}')

        if temp.get()=='amazon':
            webbrowser.open(f'https://www.amazon.com/s?k={questionField.get()}&ref=nb_sb_noss')

        if temp.get() == 'youtube':
            webbrowser.open(f'http://youtube.com/results?search_query={questionField.get()}')
        if temp.get() == 'local':
            # Main execution
            query = get_query()
            if not(query):
                messagebox.showerror("Error","After deleting the stopwords, the query became empty")
            else:
                relevant_documents = search_documents(query)

                # Print the relevant documents
                if relevant_documents:
                    resultText.delete('1.0', END) #Clear the result box
                    resultText.insert(END,"Relevant Documents:\n")
                    for doc_id, similarity in relevant_documents:
                        resultText.insert(END, f"{doc_id} Similarity score: {similarity}\n")
                        

                else:
                    resultText.delete("1.0",END)
                    resultText.insert(END,"No relevant documents found.")
    else:
        messagebox.showerror('Error','There is nothing to be searched')

def voice():
    mixer.music.load('../music1.mp3')
    mixer.music.play()
    sr=speech_recognition.Recognizer()
    with speech_recognition.Microphone() as m:
        try:
            sr.adjust_for_ambient_noise(m, duration=0.2)
            audio=sr.listen(m)
            message = sr.recognize_google(audio, language='fr-FR')
            mixer.music.load('../music2.mp3')
            mixer.music.play()
            questionField.delete(0,END)
            questionField.insert(0,message)
            search()

        except:
            pass

# Function to get stemmed words for documents
def get_stemmed_words_for_docs(docs_folder):
    stemmer = SnowballStemmer("french")
    docs_stemmed_words = {}

    for file_name in os.listdir(docs_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(docs_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                words = word_tokenize(text.lower())
                words = [word for word in words if word.isalpha() and not word in stopwords.words('french')]
                stemmed_words = [stemmer.stem(word) for word in words]
                docs_stemmed_words[file_name] = stemmed_words

    return docs_stemmed_words

# Function to calculate TF for documents
def get_tf_for_docs(docs_folder):
    stemmer = SnowballStemmer("french")
    docs_tf = {}
    for file_name in os.listdir(docs_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(docs_folder, file_name)
            with open(file_path, 'r') as file:
                text = file.read()
                words = word_tokenize(text.lower())
                stemmed_words = [stemmer.stem(word) for word in words]
                freq_dict = {}
                for word in stemmed_words:
                    if word in freq_dict:
                        freq_dict[word] += 1
                    else:
                        freq_dict[word] = 1
                docs_tf[file_name] = freq_dict
    return docs_tf

# Function to calculate IDF
def get_idf(docs_folder):
    stem_docs = get_stemmed_words_for_docs(docs_folder)
    idf = {}

    for doc_id in stem_docs:
        for word in set(stem_docs[doc_id]):
            if word in idf:
                idf[word] += 1
            else:
                idf[word] = 1

    n_docs = len(stem_docs)
    for word in idf:
        idf[word] = np.log(n_docs / idf[word])

    return idf

# Function to calculate TF-IDF weights for documents
def get_weights(docs_folder):
    stem_docs = get_stemmed_words_for_docs(docs_folder)
    idf = get_idf(docs_folder)
    weights = {}

    for doc_id in stem_docs:
        total_words = len(stem_docs[doc_id])
        tf_idf_dict = {}
        for word in stem_docs[doc_id]:
            tf = stem_docs[doc_id].count(word) / total_words
            tf_idf = tf * idf[word]
            tf_idf_dict[word] = tf_idf
        weights[doc_id] = tf_idf_dict

    return weights

# Function to read the user query
def get_query():
    query = questionField.get()
    
    stemmer = SnowballStemmer("french")  # Define the stemmer
    
    query = query.lower()
    query = word_tokenize(query)
    query = [word for word in query if word.isalpha() and word not in stopwords.words('french')]
    query = [stemmer.stem(word) for word in query]
    print(query)
    return query

# Function to search documents based on the query
def search_documents(query):
    # Get the TF-IDF weights for each document in the collection
    docs_weights = get_weights('../docs_folder')
    
    # Calculate the term frequency of each word in the query
    query_word_counts = Counter(query)
    
    # Find the maximum term frequency in the query
    max_word_count = max(query_word_counts.values())
    
    # Normalize the term frequencies in the query by the maximum term frequency
    query_word_freq = {word: count / max_word_count for word, count in query_word_counts.items()}
    
    # Calculate the TF-IDF weight for each word in the query
    query_weights = {}
    idf = get_idf('../docs_folder')
    
    for word, count in query_word_counts.items():
        if word in idf:
            tf = count / max_word_count
            tf_idf = tf * idf[word]
            query_weights[word] = tf_idf
    
    # Calculate the cosine similarity between the query and each document
    similarities = {}

    for doc_id, doc_weights in docs_weights.items():
    # Get the weights for the current document
        doc_vector = np.array(list(doc_weights.values()))

        # Create a query vector by assigning weights to the words in the document, and 0 for those not present in the document
        query_vector = np.array([query_weights.get(word, 0) for word in doc_weights.keys()])

        # Calculate the magnitude (length) of the document and query vectors
        doc_magnitude = np.linalg.norm(doc_vector)
        query_magnitude = np.linalg.norm(query_vector)

        # Check if the magnitude of the document or query vector is 0 (to avoid division by 0)
        if doc_magnitude == 0 or query_magnitude == 0:
            similarity = 0
        else:
            # Calculate the cosine similarity between the document and the query vectors
            similarity = np.dot(doc_vector, query_vector) / (doc_magnitude * query_magnitude)

        # Add the similarity score for the current document to the dictionary of similarities
        similarities[doc_id] = similarity


    # Sort the documents by relevance
    relevant_documents = [(doc_id, similarity) for doc_id, similarity in similarities.items() if similarity > 0]
    relevant_documents = sorted(relevant_documents, key=lambda x: x[1], reverse=True)

    # Return a list of relevant documents, sorted by relevance
    return relevant_documents

def build_inverted_file(docs_folder):
    # Call the get_weights function
    weights = get_weights(docs_folder)
    #Initialize the inverted file
    inverted_file = {}
    # Iterate over the weights 
    for word in weights:
        for doc_id in weights[word]:
            weight = weights[word][doc_id]
            if word in inverted_file:
                inverted_file[word][doc_id] = weight
            else:
                inverted_file[word] = {doc_id: weight}
    #Store the inverted_file in a file.txt
    with open("fich-inv1.txt", "w") as file:
        for word in sorted(inverted_file.keys()):
            file.write(word + "----")
            for doc_id in inverted_file[word]:
                weight = inverted_file[word][doc_id]
                file.write(doc_id + "-----" + str(weight) + "......")
            file.write("\n")

    return inverted_file

inverted_file = build_inverted_file('../docs_folder')

# Print the inverted file in alphabetical order
for word in sorted(inverted_file.keys()):
    print(word, "-->", inverted_file[word])


root = Tk()

root.geometry('580x400+500+250')
root.title('Spidey')

bgImg = PhotoImage(file='bg.gif')
root.iconphoto(True, bgImg)
root.config(bg='lightgrey')
root.resizable(0,0)

temp = StringVar()

style=ttk.Style()
#print(style.theme_names())
style.theme_use('default')

queryLabel= Label(root,text='Query',font=('arial',14,'bold'),bg='lightgrey')
queryLabel.grid(row=0,column=0)

#Setting the question field
questionField=Entry(root,width=30,font=('arial',14,'bold'),bd=4,relief=SUNKEN)
questionField.grid(padx=10,row=0,column=1)
#Setting the output field
outputLabel = Label(root, text='Output', font=('arial', 14, 'bold'), bg='lightgrey')
outputLabel.grid(row=1, column=0, sticky='nsew', padx=20, pady=40)
#Setting the result field
resultText = Text(root, width=50, height=10, font=('arial', 12), bd=4, relief=SUNKEN)
resultText.grid(row=2, column=0, columnspan=4, padx=20, pady=5)




buttonImg = PhotoImage(file='logoFinalForsure.png')
resizedButtonImg = buttonImg.subsample(8)
searchButton = Button(root,image=resizedButtonImg,bg='lightgrey',bd=0,cursor='hand2',activebackground='lightgrey'
                      ,command=search)
searchButton.grid(row=0,column=2)

micImg = PhotoImage(file='mic.png')
micButton = Button(root,image=micImg,bg='lightgrey',bd=0,cursor='hand2',activebackground='lightgrey'
                   ,command=voice)
micButton.grid(row=0,column=3)

googleRadioButton=ttk.Radiobutton(root,text='Google',value='google',variable=temp)
googleRadioButton.place(x=75,y=40)

duckRadioButton=ttk.Radiobutton(root,text='Duck Duck Go',value='duck',variable=temp)
duckRadioButton.place(x=200,y=40)

amzonRadioButton=ttk.Radiobutton(root,text='Amazon',value='amazon',variable=temp)
amzonRadioButton.place(x=380,y=40)

youtubeRadioButton=ttk.Radiobutton(root,text='Youtube',value='youtube',variable=temp)
youtubeRadioButton.place(x=510,y=40)

localSeacrhButton=ttk.Radiobutton(root,text='LocalSearch',value='local',variable=temp)
localSeacrhButton.place(x=210,y=80)

def enter_function(value):
    searchButton.invoke()

root.bind('<Return>',enter_function)

temp.set('google')

root.mainloop()

