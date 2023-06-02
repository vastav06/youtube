import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build
import pandas as pd
import os

# Set up YouTube API credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/credentials.json"  # Update with your own credentials file

# Load the dataset
dataset_file = "dataset.csv"
df = pd.read_csv(dataset_file)

# Encode the class labels
label_encoder = LabelEncoder()
df["Class"] = label_encoder.fit_transform(df["Class"])

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize the video titles
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df["Video Title"])

# Convert the video titles to sequences
x_train = tokenizer.texts_to_sequences(train_df["Video Title"])
x_test = tokenizer.texts_to_sequences(test_df["Video Title"])

# Pad the sequences to ensure equal length
max_length = max(max(map(len, x_train)), max(map(len, x_test)))  # Maximum sequence length
x_train = pad_sequences(x_train, maxlen=max_length, padding="post")
x_test = pad_sequences(x_test, maxlen=max_length, padding="post")

# Convert the labels to TensorFlow tensors
y_train = tf.constant(train_df["Class"].values)
y_test = tf.constant(test_df["Class"].values)


# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=max_length),
    tf.keras.layers.Conv1D(32, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(units=len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# Train the model
model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test))

# # Set up YouTube API client
# api_service_name = "youtube"
# api_version = "v3"
# youtube = build(api_service_name, api_version, developerKey="AIzaSyAkj4MbOoDdYspXUn53k4z3rSLDcae8bCg")  # Replace with your own API key

# # Function to get recommended videos based on user query
# def get_recommended_videos(query):
#     search_request = youtube.search().list(
#         q=query,
#         part="snippet",
#         maxResults=5
#     )
#     search_response = search_request.execute()
#     video_titles = [item['snippet']['title'] for item in search_response['items']]
#     return video_titles

# Get user input
search_query = input("Enter your search query: ")

# Preprocess the user query
query_sequence = tokenizer.texts_to_sequences([search_query])
query_sequence = pad_sequences(query_sequence, maxlen=max_length, padding="post")

# Make predictions
predicted_class = tf.argmax(model.predict(query_sequence), axis=-1).numpy()[0]
recommended_videos = test_df[test_df["Class"] == predicted_class]["Video Title"].head(5).tolist()

# Print recommended videos from the model
print("Recommended videos from the model:")
for video in recommended_videos:
    print(video)

# # Get recommended videos from YouTube API
# recommended_videos_api = get_recommended_videos(search_query)

# # Print recommended videos from YouTube API
# print("Recommended videos from YouTube API:")
# for video in recommended_videos_api:
#     print(video)
