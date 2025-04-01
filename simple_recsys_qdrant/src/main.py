import pandas as pd

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from utils import create_sparse_ratings_vectors
from update_qdrant import *
from algo import (
    step1_find_similar_user,
    step2_retrieve_items_from_users_ratings,
    step3_get_recommendation,
)

from utils import calculate_user_embeddings_safe

# movies embeddings
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
plot_embeddings = list(model.embed(movies_df["plot"].tolist()))

# user embeddings
user_embeddings = calculate_user_embeddings_safe(ratings_df, plot_embeddings)

user_ratings_sparse = create_sparse_ratings_vectors(ratings_df=ratings_df)


# Initialize Qdrant Client
client = QdrantClient(":memory:")  # Or your Qdrant connection details

upsert_movies_to_qdrant(
    client=client,
    collection_name="movies",
    plot_embeddings=plot_embeddings,
    movies_df=movies_df,
)
l = []  # dict to list
for x in user_embeddings.items():
    l.append(x[1])
upsert_users_to_qdrant(
    client=client, collection_name="users", user_embeddings=l, user_df=users_df
)

upsert_ratings_to_qdrant(
    client=client, collection_name="ratings", user_ratings_sparse=user_ratings_sparse
)

##########
# INIZIA ALGO
##########

user_points = step1_find_similar_user(
    client=client,
    target_user_id=users_df.user_id[0],
    collection_name="users",
    user_embeddings=user_embeddings,
)

items_from_similar_users = step2_retrieve_items_from_users_ratings(
    client=client, simalar_users=user_points
)

target_user_items = step2_retrieve_items_from_users_ratings(
    client=client,
    simalar_users=client.retrieve(
        collection_name="ratings",
        ids=[users_df.user_id[0]],
        with_vectors=True,
        with_payload=False,
    ),
)

recommendations = step3_get_recommendation(
    client=client,
    collection_name="movies",
    key_filter="index",
    target_user_items=target_user_items,
    items_from_similar_users=items_from_similar_users,
)

df = pd.DataFrame(recommendations).drop(columns=["index"], inplace=True)