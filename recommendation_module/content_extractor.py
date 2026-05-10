import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
import unicodedata
import json
from pathlib import Path

DATASET = Path("data/dataset.json")

_STEMMER = SnowballStemmer("spanish")


def stem_tokens(tokens):
    return [_STEMMER.stem(str(token)) for token in tokens if str(token).strip()]

def clean_text(text):   #limpiar texto
    text = (text or "").lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('spanish'))
    except Exception:
        tokens = text.split()
        stop_words = {'de', 'la', 'el', 'los', 'las', 'y', 'o', 'con', 'en', 'por', 'para', 'que', 'un', 'una'}
    filtered_tokens = [w for w in tokens if w and w not in stop_words]
    return stem_tokens(filtered_tokens)

def to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    if ',' in s:
        return [p.strip() for p in s.split(',') if p.strip()]
    return [s]
    
def extract_doc_feature(doc):
    url = str(doc.get("url", "")).strip()
    title = str(doc.get("title", "")).strip()
    media_type = str(doc.get("type", "")).strip()
    plot = str(doc.get("plot", "")).strip()
    genres = to_list(doc.get("genres"))
    actors = to_list(doc.get("actors"))
    directors = to_list(doc.get("director"))

    combined = " ".join([title, " ".join(genres), " ".join(directors), " ".join(actors), plot])
    tokens = clean_text(combined)

    return {
        "url": url,
        "title": title,
        "type": media_type,
        "genres": genres,
        "actors": actors,
        "director": directors,
        "plot": plot,
        "tokens": tokens,
    }
