import json
import pickle #para guardar los embeddings
from sentence_transformers import SentenceTransformer

def build_embeddings():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') #cargamos el modelo multilingue
    urls = []
    text = []
    with open('data/movies.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for mov in data:
        urls.append(mov['url'])  #listas con las urls
        text.append(mov['title'] + " " + mov['plot']) #listas con la info, o sea, el titulo y el argumento
    embeddings = model.encode(text) # creamos la matriz de embeddings con el metodo encode de model, donde cada vector representa una url
    dic = {  #lo guardamos todo en un diccionario para despues exportarlo como un archivo pkl
        "urls": urls,
        "embeddings": embeddings
    }
    with open('bd/movies_vectors.pkl', 'wb') as f:
        pickle.dump(dic, f)

        