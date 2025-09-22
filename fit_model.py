import pandas as pd

import toolbox as tbx

df = pd.read_csv("./stash/openalex_llm_social_02072025.csv", 
    usecols=["title", "abstract", "topics.display_name", "language","id"])

# filter off lines where language is not english
# df["language"].unique() 
# >>> array(['en', 'ca', nan, 'es', 'pt', 'hu', 'ru'], dtype=object)
df = df.loc[df["language"] == "en", :] # approx 99% of the lines are kept

umap_model_parameters = {
    "n_neighbors"   : 10,
    "n_components"  : 2,
    "min_dist"      : 0.1
}

hdbscan_model_parameters = {
    "hdbscan_min_cluster_size" : 10,
}

topic_model_parameters = {
    "nr_topics"         : None, 
    "min_topic_size"    : 10
}


topic_model = tbx.setup(umap_model_parameters, hdbscan_model_parameters, topic_model_parameters)
topics, _ = topic_model.fit_transform(**tbx.fetch_documents_and_embedding())

pd.DataFrame({'topics' : topics}).\
    to_csv("./stash/topics.csv", index = False)