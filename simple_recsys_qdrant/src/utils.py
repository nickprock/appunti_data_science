import random
import pandas as pd
import numpy as np
import uuid
from faker import Faker

def generate_random_movie_dataset(num_movies=100):
    """Generates a more diverse random dataset of movies.

    Args:
        num_movies: The number of movies to generate.

    Returns:
        A pandas DataFrame containing the movie data.
    """

    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Horror", "Thriller",
              "Romance", "Animation", "Mystery", "Adventure", "Crime", "Documentary",
              "Family", "History", "Music", "War", "Western"]
    plot_starters = [
        "A group of ", "A lone ", "An unlikely team of ", "The quest for a ",
        "In a world where ", "When a ", "The mystery of a ", "A forbidden ",
        "The legend of a ", "The last days of a ", "The rise and fall of a ",
        "A battle between ", "The search for ", "The secret life of ", "The truth behind "
    ]
    plot_subjects = [
        "friends", "detective", "lovers", "robots", "sorcerer", "ghost",
        "thieves", "artist", "king", "city", "empire", "armies", "treasure",
        "clowns", "ancient prophecy"
    ]
    plot_settings = [
        "small town", "bustling city", "distant galaxy", "parallel universe",
        "haunted mansion", "desert planet", "underwater city", "medieval kingdom",
        "futuristic metropolis", "forgotten island", "time-traveling train",
        "virtual reality", "dream world", "post-apocalyptic wasteland",
        "secret laboratory"
    ]
    plot_modifiers = [
        "uncovers a conspiracy", "faces their past", "fights for survival",
        "falls in love", "discovers a hidden power", "seeks revenge",
        "tries to save the world", "learns a shocking truth",
        "embarks on a dangerous mission", "confronts their fears",
        "makes an impossible choice", "challenges destiny",
        "finds unexpected allies", "changes everything", "is not what it seems"
    ]

    movie_data = []
    for i in range(1, num_movies + 1):
        title_parts = [
            random.choice(["The", "A", "An"]),
            random.choice(["Shadow", "Secret", "Lost", "Hidden", "Forgotten"]),
            random.choice(["City", "World", "Kingdom", "Empire", "Legacy"]),
            random.choice(["of", "and the", "vs.", "in"]),
            random.choice(["Magic", "Steel", "Time", "Destiny", "Dreams"])
        ]
        title = " ".join(title_parts) + random.choice(["", ": Origins", ": Reckoning", " Begins", " Returns"])

        genre = random.choice(genres)
        plot = random.choice(plot_starters) + random.choice(plot_subjects) + " " + \
               random.choice(plot_modifiers) + " in a " + random.choice(plot_settings) + "."

        movie_data.append({"movieId": str(uuid.uuid4()), "title": title, "genre": genre, "plot": plot, "index":i})

    return pd.DataFrame(movie_data)



def generate_random_user_dataset(num_users=10):
    """Generates a random dataset of users.

    Args:
        num_users: The number of users to generate.

    Returns:
        A pandas DataFrame containing the user data.
    """

    fake = Faker()  # Initialize Faker for name generation
    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Horror", "Thriller",
              "Romance", "Animation", "Mystery", "Adventure", "Crime"]  # More genres

    user_data = []
    for i in range(1, num_users + 1):
        gender = random.choice(["Male", "Female", "Other"])
        if gender == "Male":
            first_name = fake.first_name_male()
            last_name = fake.last_name()
        elif gender == "Female":
            first_name = fake.first_name_female()
            last_name = fake.last_name()
        else:
            first_name = fake.first_name()
            last_name = fake.last_name()

        age = random.randint(16, 65)  # Age from 16 to 65
        preferred_genre = random.choice(genres)

        user_data.append({
            "user_id": str(uuid.uuid4()),
            "first_name": first_name,
            "last_name": last_name,
            "age": age,
            "gender": gender,
            "preferred_genre": preferred_genre
        })

    return pd.DataFrame(user_data)

def generate_consistent_ratings(user_ids, movie_ids, num_ratings=100):
    """
    Generates a dataset of user-movie ratings with some consistency, using provided user and movie IDs.

    Args:
        user_ids:   A list or set of user IDs (strings).
        movie_ids:  A list or set of movie IDs (strings).
        num_ratings: The total number of ratings to generate.

    Returns:
        A pandas DataFrame containing the rating data.
    """

    ratings_data = []
    num_users = len(user_ids)
    num_movies = len(movie_ids)

    # Simulate user preferences (simplified)
    user_preferences = {}
    for user_id in user_ids:
        # Each user has some "preferred" genres (for variety)
        num_preferred_genres = random.randint(1, 3)
        preferred_genres = random.sample(range(10), num_preferred_genres)  # Assuming 10 genres
        user_preferences[user_id] = preferred_genres

    # Create a mapping of movie_id to an integer representation for genre assignment
    movie_id_to_int = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    for _ in range(num_ratings):
        user_id = random.choice(list(user_ids))  # Ensure we pick from provided IDs
        movie_id = random.choice(list(movie_ids))  # Ensure we pick from provided IDs

        # Simulate genre-based rating
        # (This is where the "consistency" comes in)
        movie_genre = movie_id_to_int[movie_id] % 10  # Use the integer representation for genre
        if movie_genre in user_preferences.get(user_id, []):
            rating = random.choices([4, 5, 3], weights=[0.5, 0.3, 0.2])[0]  # Higher chance of high ratings
        else:
            rating = random.choices([1, 2, 3], weights=[0.2, 0.3, 0.5])[0]  # Higher chance of low ratings

        ratings_data.append({"user_id": user_id, "movie_id": movie_id, "rating": rating})
        df = pd.DataFrame(ratings_data)

    return df


def calculate_user_embeddings_safe(ratings_df: pd.DataFrame, plot_embeddings: np.ndarray):
    """
    Calculates user embeddings from ratings data, handling potential NaN issues.

    Args:
        ratings_df: DataFrame with 'user_id', 'movie_id', and 'rating' columns.
        plot_embeddings: NumPy array where each row represents the embedding of a movie.
                       It's assumed that the index of the row corresponds to the movie_id.

    Returns:
        A dictionary where keys are unique user_ids and values are their calculated embeddings.
        Returns an empty dictionary if any error occurs.
    """

    embeddings = {}  # Changed to a dictionary for clarity and efficiency
    user_idx = set()  # Use a set for faster membership checking

    for user_id in ratings_df["user_id"].unique():  # Iterate over unique user IDs
        try:
            movie_ids = ratings_df.loc[ratings_df["user_id"] == user_id, "movie_id"].values
            ratings = ratings_df.loc[ratings_df["user_id"] == user_id, "scaled_rating"].values
            emb = np.zeros_like(plot_embeddings[0], dtype=np.float32)  # Initialize with zeros and float32

            for i, movie_id in enumerate(movie_ids):
                # Safe access to plot_embeddings (handle potential out-of-bounds)
                if 0 <= movie_id < len(plot_embeddings):  # Basic bounds check
                    if not np.any(np.isnan(plot_embeddings[movie_id])) and not np.isnan(ratings[i]):
                        emb += plot_embeddings[movie_id] * ratings[i]
                    elif np.any(np.isnan(plot_embeddings[movie_id])):
                        print(f"Warning: NaN values in plot_embeddings for movie_id {movie_id}, skipping.")
                    elif np.isnan(ratings[i]):
                        print(f"Warning: NaN value in rating for user {user_id} and movie {movie_id}, skipping.")
                else:
                    print(f"Warning: movie_id {movie_id} out of bounds, skipping.")

                if np.any(np.isnan(emb)):
                    print(f"Warning: NaN encountered in user embedding for user {user_id} during calculation.")
                    emb = np.zeros_like(plot_embeddings[0], dtype=np.float32)  # Reset emb
                    break  # Stop processing ratings for this user

            embeddings[user_id] = emb  # Store the calculated embedding

        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            return {}  # Return an empty dictionary to signal an error

    return embeddings

def create_sparse_ratings_vectors(ratings_df):
    """
    Creates a dictionary where keys are user_ids and values are tuples
    containing lists of movie_ids and scaled_ratings.

    Args:
        ratings_df: DataFrame with 'user_id', 'movie_id', and 'scaled_rating' columns.

    Returns:
        A dictionary where keys are user_ids and values are tuples
        (list of movie_ids, list of scaled_ratings).
    """

    user_ratings_sparse = {}
    for i in range(ratings_df.shape[0]):
        user_id = ratings_df.user_id[i]
        movie_id = ratings_df.movie_id[i]
        scaled_rating = ratings_df.scaled_rating[i]

        if user_id not in user_ratings_sparse:
            user_ratings_sparse[user_id] = ([], [])  # Initialize as a tuple of two empty lists

        try:
            if str(movie_id) not in user_ratings_sparse[user_id][0]:
                user_ratings_sparse[user_id][0].append(str(movie_id))  # Store movie_id as string
                user_ratings_sparse[user_id][1].append(scaled_rating)
        except ValueError:
            print(f"Warning: movie_id '{movie_id}' is not a valid type. Skipping this rating.")
            continue



    return user_ratings_sparse
