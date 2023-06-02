

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
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


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')


# # Create a Flask application
# app = Flask(__name__)

# # Define the route for the home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# Define the route for handling form submission




@app.route('/recommend', methods=['GET','POST'])
def recommend():
    search_query = request.form['search_query']  # Get the input from the form
    query_sequence = tokenizer.texts_to_sequences([search_query])
    query_sequence = pad_sequences(query_sequence, maxlen=max_length, padding="post")
    predicted_class = tf.argmax(model.predict(query_sequence), axis=-1).numpy()[0]
    recommended_videos = test_df[test_df["Class"] == predicted_class]["Video Title"].head(5).tolist()
    return render_template('recommend.html', videos=recommended_videos)

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8080, debug=True)


