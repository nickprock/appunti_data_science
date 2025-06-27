import pandas as pd
import umap


class UMAPGowerEmbedder:
    """
    UMAP-based embedder that simulates Gower Distance by combining
    numerical and categorical features with different metrics.
    """
    
    def __init__(self, n_components=2, intersection=False, random_state=12, n_neighbors_categorical=250):
        """
        Initialize the UMAP Gower Distance embedder.
        
        Parameters:
        -----------
        n_components : int, default=2
            Number of dimensions for the embedding
        intersection : bool, default=False
            If True, uses intersection (multiplication) of embeddings (resembles numerical more)
            If False, uses union (addition) of embeddings (resembles categorical more)
        random_state : int, default=12
            Random state for reproducibility
        n_neighbors_categorical : int, default=250
            Number of neighbors for categorical UMAP
        """
        self.n_components = n_components
        self.intersection = intersection
        self.random_state = random_state
        self.n_neighbors_categorical = n_neighbors_categorical
        
        # UMAP models will be stored here after fitting
        self.numerical_umap = None
        self.categorical_umap = None
        self.combined_embedding = None
        self.is_fitted = False
        
        # Store column information for transform
        self.numerical_columns = None
        self.categorical_columns = None
        self.categorical_encoder_columns = None
    
    def fit(self, df):
        """
        Fit the UMAP embedders on the data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with mixed numerical and categorical features
            
        Returns:
        --------
        self : UMAPGowerEmbedder
            Returns self for method chaining
        """
        # Separate numerical and categorical features
        numerical = df.select_dtypes(exclude='object')
        categorical = df.select_dtypes(include='object')
        
        # Store column information
        self.numerical_columns = numerical.columns.tolist()
        self.categorical_columns = categorical.columns.tolist()
        
        # One-hot encode categorical features
        categorical_encoded = pd.get_dummies(categorical)
        self.categorical_encoder_columns = categorical_encoded.columns.tolist()
        
        # Initialize and fit UMAP models
        self.numerical_umap = umap.UMAP(
            metric="manhattan", 
            random_state=self.random_state,
            n_components=self.n_components
        ).fit(numerical)
        
        self.categorical_umap = umap.UMAP(
            metric='dice', 
            n_neighbors=self.n_neighbors_categorical,
            n_components=self.n_components, 
            random_state=self.random_state
        ).fit(categorical_encoded)
        
        # Combine embeddings
        if self.intersection:
            # Intersection will resemble the numerical embedding more
            self.combined_embedding = self.numerical_umap * self.categorical_umap
        else:
            # Union will resemble the categorical embedding more
            self.combined_embedding = self.numerical_umap + self.categorical_umap
        
        self.is_fitted = True
        return self
    
    def transform(self, df):
        """
        Transform new data using the fitted embedders.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with the same structure as training data
            
        Returns:
        --------
        numpy.ndarray
            UMAP embedding of the input data
        """
        if not self.is_fitted:
            raise ValueError("The embedder must be fitted before transforming data. Call 'fit' first.")
        
        # Separate numerical and categorical features
        numerical = df.select_dtypes(exclude='object')
        categorical = df.select_dtypes(include='object')
        
        # Verify column consistency
        if numerical.columns.tolist() != self.numerical_columns:
            raise ValueError("Numerical columns in transform data don't match training data")
        if categorical.columns.tolist() != self.categorical_columns:
            raise ValueError("Categorical columns in transform data don't match training data")
        
        # One-hot encode categorical features
        categorical_encoded = pd.get_dummies(categorical)
        
        # Ensure all categorical columns from training are present
        for col in self.categorical_encoder_columns:
            if col not in categorical_encoded.columns:
                categorical_encoded[col] = 0
        
        # Reorder columns to match training order
        categorical_encoded = categorical_encoded[self.categorical_encoder_columns]
        
        # Transform using fitted models
        numerical_transformed = self.numerical_umap.transform(numerical)
        categorical_transformed = self.categorical_umap.transform(categorical_encoded)
        
        # Combine embeddings using the same method as in fit
        if self.intersection:
            combined_transformed = numerical_transformed * categorical_transformed
        else:
            combined_transformed = numerical_transformed + categorical_transformed
        
        return combined_transformed
    
    def fit_transform(self, df):
        """
        Fit the embedder and transform the data in one step.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with mixed numerical and categorical features
            
        Returns:
        --------
        numpy.ndarray
            UMAP embedding of the input data
        """
        self.fit(df)
        return self.combined_embedding.embedding_
    
    def get_embedding(self):
        """
        Get the embedding from the fitted model.
        
        Returns:
        --------
        numpy.ndarray
            The combined UMAP embedding
        """
        if not self.is_fitted:
            raise ValueError("The embedder must be fitted before getting the embedding.")
        return self.combined_embedding.embedding_
