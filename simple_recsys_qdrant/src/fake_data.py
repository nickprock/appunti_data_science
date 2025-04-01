from utils import generate_random_movie_dataset, generate_random_user_dataset, generate_consistent_ratings

# Generate the dataset
movies_df = generate_random_movie_dataset(num_movies=10)

# Print the first few rows to see the data
# print(movies_df.head())

# Save to CSV (optional)
movies_df.to_csv("../dati/random_movies.csv", index=False)


users_df = generate_random_user_dataset(num_users=10)

# print(users_df.head())
# print(users_df.info())
users_df.to_csv("../dati/random_users.csv", index=False)


user_ids = users_df.user_id.unique().tolist()
movie_ids = movies_df.index.unique().tolist()

ratings_df = generate_consistent_ratings(user_ids, 
                                         movie_ids, 
                                         num_ratings=100)

print(ratings_df.head())
print(ratings_df.info())  # Show data types and counts
ratings_df["scaled_rating"] = ratings_df["rating"].apply(lambda x: x / 5)
ratings_df.to_csv("../dati/consistent_ratings.csv", index=False)