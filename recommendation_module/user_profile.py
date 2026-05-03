import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import unicodedata

from recommendation_module.content_extractor import stem_tokens

class User:
    def __init__(self, user_id):
        self.user_id = user_id          #id del usuario
        self.search_history = []        #lista con todo su historial de busqueda
        self.clicked_urls = []          #lista con todas las URLs que ha hecho clic
        self.viewed_urls = []           #lista con todas las URLs que ha visto aunque no haya hehco clic
        self.genre_preferences = {}     #diccionario con las preferencias de género del usuario
        self.type_preferences = {}      #diccionario con las preferencias de tipo del usuario(serie, pelicula)
        self.term_preferences = {}      #diccionario con las palabras que el usurio mas prefiere

    def register_search(self, query):    #registra la busqueda
        if query.strip() == "":
            return
        self.search_history.append(query)
        
        text = query.lower()                                                  #convertir todo en minusculas
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8','ignore')      #eliminar tildes
        text = re.sub(r'[^a-z\s]', '', text)                                 #eliminar todo lo que no sea letras de la a a la z
        tokens = word_tokenize(text)                                         #convertir el texto en una lista de tokens
        stop_words = set(stopwords.words('spanish'))                         #set con las palabras que no aportan nada como articulos
        filtered_tokens = [w for w in tokens if w not in stop_words]         #nos quedamos con todas las palabras menos las stop words
        stems = stem_tokens(filtered_tokens)

        for t in stems:             #guarda las palabras en el dictionario de preferncias de termino con su peso
            if t not in self.term_preferences:
                self.term_preferences[t] = 1
            else:
                self.term_preferences[t] += 1

    def register_click(self, url):      #metodo que guarda las URLs en las que el usuario ha hecho clic
        if url.strip() == "":
            return
        url = url.lower()
        if url in self.clicked_urls:
            return
        self.clicked_urls.append(url)

    def register_view(self, url):  #metodo que guarda las URLs en las que el usuario ha visto
        if url.strip() == "":
            return
        url = url.lower()
        if url in self.viewed_urls:
            return
        self.viewed_urls.append(url)

    def add_genre_preference(self, genre):   #metodo que agrega una preferencia de género
        genre = genre.strip().lower()
        if genre == "":
            return
        if genre not in self.genre_preferences:
            self.genre_preferences[genre] = 1
        else:
            self.genre_preferences[genre] += 1

    def add_type_preference(self, media_type):      #metodo que agrega una preferencia de tipo
        media_type = media_type.strip().lower()
        if media_type == "":
            return
        if media_type not in self.type_preferences:
            self.type_preferences[media_type] = 1
        else:
            self.type_preferences[media_type] += 1

    def to_dict(self):                   #metodo que convierte el perfil del usuario en un diccionario
        return {
            "user_id": self.user_id,
            "search_history": list(self.search_history),
            "clicked_urls": list(self.clicked_urls),
            "viewed_urls": list(self.viewed_urls),
            "genre_preferences": dict(self.genre_preferences),
            "type_preferences": dict(self.type_preferences),
            "term_preferences": dict(self.term_preferences),
        }

    @classmethod                             #metodo que crea un perfil de usuario a partir de un diccionario
    def from_dict(cls, data):
        user = cls(data.get("user_id", "anonymous"))
        user.search_history = list(data.get("search_history", []))
        user.clicked_urls = list(data.get("clicked_urls", []))
        user.viewed_urls = list(data.get("viewed_urls", []))
        user.genre_preferences = dict(data.get("genre_preferences", {}))
        user.type_preferences = dict(data.get("type_preferences", {}))
        user.term_preferences = dict(data.get("term_preferences", {}))
        return user


