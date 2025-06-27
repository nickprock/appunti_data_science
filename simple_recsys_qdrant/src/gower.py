import pandas as pd
import umap

def umap_embed(df, n_components=2, intersection=False):
    """
    Simulate Gower Distance using UMAP
    """
    numerical = df.select_dtypes(exclude='object')
    categorical = df.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)

    #Embedding numerical & categorical
    fit1 = umap.UMAP(metric = "manhattan", random_state=12,
                   n_components=n_components).fit(numerical)
  
    fit2 = umap.UMAP(metric='dice', 
                   n_neighbors=250,
                   n_components=n_components, random_state=12).fit(categorical)

    # intersection will resemble the numerical embedding more.
    if intersection:
        embedding = fit1 * fit2

    # union will resemble the categorical embedding more.
    else:
        embedding = fit1 + fit2

    umap_embedding = embedding.embedding_
    return umap_embedding
