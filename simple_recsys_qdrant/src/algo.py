from qdrant_client import QdrantClient, models

def step1_find_similar_user(client: QdrantClient, target_user_id : str, collection_name: str,  user_embeddings : dict, top_k : int = 5):
    query = user_embeddings[target_user_id]
    return client.query_points(
        collection_name=collection_name,
        query=query,
        limit=top_k
    ).points

def step2_retrieve_items_from_users_ratings(client: QdrantClient, simalar_users:list, collection_name: str = "ratings", vector_name: str = "user_ratings"):
    retrived_users_id = [p.id for p in simalar_users]
    records = client.retrieve(collection_name=collection_name, ids=retrived_users_id, with_vectors=True, with_payload=False)
    items = []
    for r in records:
        for i in range(len(r.vector[vector_name].indices)):
            if r.vector[vector_name].values[i]>0.5:
                items.append(r.vector[vector_name].indices[i])
    return list(set(items))

def step3_get_recommendation(client: QdrantClient, collection_name:str, key_filter:str, target_user_items:list, items_from_similar_users:list, limit:int=5):
    target_user_items_info = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=key_filter,
                    match=models.MatchAny(any=target_user_items),
                )
            ]
        ),
        with_payload=True,
        with_vectors=True,
    )[0]

    prefetches = []

    for p in target_user_items_info:
        prefetches.append(models.Prefetch(
                query=p.vector,
                filter=models.Filter(must=[models.FieldCondition(
                    key=key_filter,
                    match=models.MatchAny(any=items_from_similar_users)
                    )]),
                limit=limit,
            ))
    
    retrieved_points = client.query_points(
        collection_name=collection_name,
        prefetch=prefetches,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
    ).points

    payloads = []
    for p in retrieved_points:
        payloads.append(p.payload)
    return payloads
