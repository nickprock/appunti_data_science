from qdrant_client import QdrantClient, models
import pandas as pd

client = QdrantClient(":memory:")  # Or your Qdrant connection details

# Movies
def upsert_movies_to_qdrant(client: QdrantClient, collection_name: str, plot_embeddings:list, movies_df: pd.DataFrame):
    """
    Upserts movies vectors into a Qdrant collection.

    Args:
        client: QdrantClient instance.
        collection_name: Name of the Qdrant collection.
        plot_embeddings: list of embeddings.
        movies_df: pandas dataframe for payloads
    """

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(plot_embeddings[0]), 
                distance=models.Distance.COSINE),
        )
    
    points = []
    for index, row in movies_df.iterrows():
        point = models.PointStruct(
            id=row["movieId"],  # Use movieId as the unique identifier
            vector=plot_embeddings[index],
            payload={
                "title": row["title"],
                "genre": row["genre"],
                "plot": row["plot"],
                "index": row["index"],
            }
        )
        points.append(point)
    
    client.upsert(
    collection_name=collection_name,
    wait=True,  # Wait for the operation to complete
    points=points
    )

    print(f"Movie data with plot embeddings loaded into Qdrant collection '{collection_name}'.")


# users
def upsert_users_to_qdrant(client: QdrantClient, collection_name: str, user_embeddings:list, user_df: pd.DataFrame):
    """
    Upserts movies vectors into a Qdrant collection.

    Args:
        client: QdrantClient instance.
        collection_name: Name of the Qdrant collection.
        user_embeddings: list of embeddings.
        user_df: pandas dataframe for payloads
    """

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(user_embeddings[0]), 
                distance=models.Distance.COSINE),
        )

    points = []
    for index, row in user_df.iterrows():
        point = models.PointStruct(
            id=row["user_id"],
            vector=user_embeddings[index],
            payload={
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "age": row["age"],
                "gender": row["gender"],
                "preferred_genre": row["preferred_genre"],
            }
        )
        points.append(point)
    
    client.upsert(
    collection_name=collection_name,
    wait=True,  # Wait for the operation to complete
    points=points
    )

    print(f"User data with calculated embeddings loaded into Qdrant collection '{collection_name}'.")

def upsert_ratings_to_qdrant(client: QdrantClient, collection_name: str, user_ratings_sparse: dict):
    """
    Upserts user ratings as sparse vectors into a Qdrant collection.

    Args:
        client: QdrantClient instance.
        collection_name: Name of the Qdrant collection.
        user_ratings_sparse: Dictionary of user ratings in sparse vector format.
    """
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={},
            sparse_vectors_config={
                "user_ratings": models.SparseVectorParams()
            }
        )

    points = []
    for user_id, sparse_data in user_ratings_sparse.items():
        point = models.PointStruct(
            id=user_id,  # Qdrant requires string IDs
            # Remove the 'vector' field and only include 'sparse_vector'
            vector={
                "user_ratings": models.SparseVector(
                    indices=sparse_data[0],
                    values=sparse_data[1]
                )
            },
            # payload={
            #     "key": user_id,
            # }
        )
        points.append(point)

    client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )
    print(f"Ratings data with sparse vectors loaded into Qdrant collection '{collection_name}'.")