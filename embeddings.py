import pandas as pd
from numpy import logical_not
from sentence_transformers import SentenceTransformer
from torch import Tensor, save

# model = SentenceTransformer("google-bert/bert-base-uncased")

df = pd.read_csv("openalex_llm_social_02072025.csv", 
    usecols=["title", "abstract", "topics.display_name", "language","id"])

df = df.loc[df["language"] == "en", ]
# drop na
df = df.loc[logical_not(df["abstract"].isna()), :]

# TESTING
df = df.iloc[:100]

# sentences = df["abstract"].to_list()
# print(len(sentences))
# print("Wrong format sentences : ", [sentence for sentence in sentences if not(isinstance(sentence, str))])

# embeddings = Tensor(model.encode(sentences))

# save(embeddings, "embeddings.pt")


df["abstract"].to_csv("abstracts.csv",index = False)