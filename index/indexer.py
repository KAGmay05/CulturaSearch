import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re
import unicodedata

nltk.download('punkt_tab')
nltk.download('stopwords')

def clean_text(text):
    text = text.lower()  #convertir todo en minusculas
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8','ignore') #eliminar tildes
    text = re.sub(r'[^a-z\s]', '', text) #eliminar todo lo que no sea letras de la a a la z
    tokens = word_tokenize(text) #convertir el texto en una lista de tokens
    stop_words = set(stopwords.words('spanish')) #set con las palabras que no aportan nada como articulos
    filtered_tokens = [w for w in tokens if w not in stop_words] #nos quedamos con todas las palabras menos las stop words
    stemmer = SnowballStemmer('spanish')  
    stems = [stemmer.stem(w) for w in filtered_tokens] #obtiene la raiz de todas las palabras
    return stems
    
def build_index(path):
    with open(path, 'r', encoding='utf-8') as f: #abre el json con las peliculas y series
        data = json.load(f)
    index = {}   #dictionario donde la llave es el token y el valor es una lista de las url donde aparece
    for mov in data:
        url = mov['url'] 
        title = str(mov.get('title', "")) #todo sea string, ni listas ni vacio 
        plot = str(mov.get('plot', ""))
        genres = " ".join(mov['genres']) if isinstance(mov.get('genres'), list) else str(mov.get('genres', ""))
        actors = " ".join(mov['actors']) if isinstance(mov.get('actors'), list) else str(mov.get('actors', ""))
        text = f"{title} {genres} {plot} {actors}" #nos quedamos solo el titulo, genero, trama, director y actores
        tokens = clean_text(text) #luego aplicamos la funcion clean text a todo ese texto
        for tok in tokens:  #agregamos cada palabra al dicionario, agragamos tambien la cantidad de veces que aparece en una url
            if tok not in index:
                index[tok] = {}
            if url not in index[tok]:
                index[tok][url] = 1
            else:
                index[tok][url] += 1
    with open('index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=4) #guardamos index en un json
    return index