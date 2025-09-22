from bertopic import BERTopic
from hdbscan import HDBSCAN
from numpy import ndarray
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import stopwordsiso as stopwords
from torch import load
from umap import UMAP

from .customLemmatizerClass import CustomLemmaTokenizer

def create_umap_model(n_neighbors : int,n_components : int,min_dist : float ,
        metric : str = "cosine") -> UMAP:
    ''''''
    return UMAP(
        n_neighbors  = n_neighbors,
        n_components = n_components,
        min_dist     = min_dist,
        metric       = metric
    ) 

def create_hdbscan_model(hdbscan_min_cluster_size : int, 
        metric : str = "euclidean") -> HDBSCAN:
    ''''''
    return HDBSCAN(
        min_cluster_size = hdbscan_min_cluster_size,
        metric           = metric,
        prediction_data  = True,
    ) 

def create_topic_model(nr_topics : int, min_topic_size : int, umap_model : UMAP, 
        hdbscan_model : HDBSCAN) -> BERTopic:
    ''''''
    vectorizer_model = CountVectorizer(
        stop_words=list(stopwords.stopwords("en")),
        tokenizer=CustomLemmaTokenizer()
    )

    return BERTopic(
        language            = "en",
        vectorizer_model    = vectorizer_model,
        nr_topics           = nr_topics,
        min_topic_size      = min_topic_size,
        umap_model          = umap_model,
        hdbscan_model       = hdbscan_model,
    )

def setup(umap_parameters : dict, hdbscan_parameters : dict, 
        bertopic_parameters : dict) -> BERTopic:
    ''''''
    bertopic_parameters = {
        "umap_model" : create_umap_model(**umap_parameters),
        "hdbscan_model" : create_hdbscan_model(**hdbscan_parameters),
        **bertopic_parameters
    }
    return create_topic_model(**bertopic_parameters)

def fetch_documents_and_embedding()->dict[str : list[str]|ndarray]:
    docs = pd.read_csv("abstracts.csv")["abstract"].to_list()
    embs = load("embeddings.pt", weights_only=True).numpy()
    return docs, embs

