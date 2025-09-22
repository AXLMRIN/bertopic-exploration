TESTINGS = True
# ==============================================================================
import pandas as pd
import toolbox as tbx
from itertools import product
from tqdm import tqdm

# Ranges
umap_n_neighbors_range = [3, 7, 11, 15, 19]
umap_n_components_range = [2, 5, 10]
umap_min_dist_range = [0, 0.1, 0.5]
hdbscan_min_cluster_size_range = [5, 10, 15, 20]
bertopic_nr_topics_range = [None, 10, 20]
bertopic_min_topic_size_range = [5, 10, 20]

all_combinations = list(product(
    umap_n_neighbors_range,umap_n_components_range,umap_min_dist_range,
    hdbscan_min_cluster_size_range,bertopic_nr_topics_range,
    bertopic_min_topic_size_range))

# Open data
df = pd.read_csv("./stash/openalex_llm_social_02072025.csv", 
    usecols=["title", "abstract", "topics.display_name", "language","id"])
df = df.loc[(df["language"] == "en")&(~df["abstract"].isna()), :] # approx 75% of the lines are kept

# Generate embeddings
tbx.generate_embeddings(df, testing=TESTINGS)

results = []
for combination in tqdm(all_combinations):
    n_neighbors, n_components, min_dist, hdbscan_min_cluster_size, nr_topics, min_topic_size = combination
    umap_model_parameters = {
        "n_neighbors"   : n_neighbors,
        "n_components"  : n_components,
        "min_dist"      : min_dist,
    }

    hdbscan_model_parameters = {
        "hdbscan_min_cluster_size" : hdbscan_min_cluster_size,
    }

    topic_model_parameters = {
        "nr_topics"         : nr_topics, 
        "min_topic_size"    : min_topic_size,
    }
    topic_model, lemmatizer = tbx.setup(umap_model_parameters, hdbscan_model_parameters, topic_model_parameters)
    topics, _ = topic_model.fit_transform(**tbx.fetch_documents_and_embedding())
    tbx.save_topics(topics, topic_model.get_topic_info(), lemmatizer)
    results += [tbx.measure_performances()]
    
    if TESTINGS & (len(results) > 2): break

pd.DataFrame(results).to_csv("./stash/results.csv", index = False)