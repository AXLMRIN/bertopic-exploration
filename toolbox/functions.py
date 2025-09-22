from bertopic import BERTopic
from hdbscan import HDBSCAN
from numpy import ndarray, logical_not
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import stopwordsiso as stopwords
from torch import load, save, Tensor
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
        hdbscan_model : HDBSCAN) -> tuple[BERTopic, CustomLemmaTokenizer]:
    ''''''
    lemmatizer = CustomLemmaTokenizer()
    vectorizer_model = CountVectorizer(
        stop_words   = list(stopwords.stopwords("en")),
        tokenizer    = lemmatizer 
    )

    topic_model =  BERTopic(
        language            = "en",
        vectorizer_model    = vectorizer_model,
        nr_topics           = nr_topics,
        min_topic_size      = min_topic_size,
        umap_model          = umap_model,
        hdbscan_model       = hdbscan_model,
    )
    return topic_model, lemmatizer

def setup(umap_parameters : dict, hdbscan_parameters : dict, 
        bertopic_parameters : dict) -> tuple[BERTopic, CustomLemmaTokenizer]:
    ''''''
    bertopic_parameters = {
        "umap_model" : create_umap_model(**umap_parameters),
        "hdbscan_model" : create_hdbscan_model(**hdbscan_parameters),
        **bertopic_parameters
    }
    return create_topic_model(**bertopic_parameters)

def fetch_documents_and_embedding(as_tuple : bool = False)->dict[str : list[str]|ndarray]:
    docs = pd.read_csv("./stash/abstracts.csv")["abstract"].to_list()
    embs = load("./stash/embeddings.pt", weights_only=True).numpy()
    if as_tuple : return docs, embs
    else : return {"documents" : docs, "embeddings" : embs}

def generate_embeddings(testing : bool = False):
    df = pd.read_csv("./stash/openalex_llm_social_02072025.csv", 
        usecols=["title", "abstract", "topics.display_name", "language","id"])

    df = df.loc[df["language"] == "en", ]
    # drop na
    df = df.loc[logical_not(df["abstract"].isna()), :]

    if testing : df = df.iloc[:100]

    sentences = df["abstract"].to_list()
    wrong_format_sentences = [sentence for sentence in sentences if not(isinstance(sentence, str))]
    if len(wrong_format_sentences)>0:
        print("Wrong format sentences : ", wrong_format_sentences)

    model = SentenceTransformer("google-bert/bert-base-uncased")
    embeddings = Tensor(model.encode(sentences))
    save(embeddings, "./stash/embeddings.pt")


    df["abstract"].to_csv("./stash/abstracts.csv",index = False)