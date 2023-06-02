# import os
# import csv
# import json
# import googleapiclient.discovery
# from googleapiclient.errors import HttpError

# # Set up YouTube API client
# api_service_name = "youtube"
# api_version = "v3"
# api_key = "AIzaSyAkj4MbOoDdYspXUn53k4z3rSLDcae8bCg"
# youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=api_key)

# # Define the five classes
# classes = ["machine learning", "blockchain","quant trading","cyber security","web3","finance","consulting","web development", "iot", "defi","crypto","Stocks","investment banking"]

# # Define the dataset file path
# dataset_file = "dataset.csv"

# # Create the dataset file and write the header
# with open(dataset_file, mode="w", newline="", encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Video Title", "Video Description", "Class"])

# # Iterate through each class
# for class_name in classes:
#     # Search for videos using the class as the query
#     try:
#         search_response = youtube.search().list(
#             part="snippet",
#             q=class_name,
#             type="video",
#             maxResults=10000
#         ).execute()

#         # Iterate through the search results
#         for search_result in search_response.get("items", []):
#             video_id = search_result["id"]["videoId"]
#             video_title = search_result["snippet"]["title"]
#             video_description = search_result["snippet"]["description"]

#             # Write the video details and class to the dataset file
#             with open(dataset_file, mode="a", newline="", encoding="utf-8") as file:
#                 writer = csv.writer(file)
#                 writer.writerow([video_title, video_description, class_name])

#     except HttpError as e:
#         print("An HTTP error occurred:")
#         print(e)

# print("Dataset created successfully!")


import os
import csv
import json
import googleapiclient.discovery
from googleapiclient.errors import HttpError

# Set up YouTube API client
api_service_name = "youtube"
api_version = "v3"
api_key = "AIzaSyAkj4MbOoDdYspXUn53k4z3rSLDcae8bCg"
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=api_key)

# Define the five classes
classes = ["machine learning", "blockchain", "quant trading", "cyber security", "web3", "finance", "consulting",
           "web development", "iot", "defi", "crypto", "Stocks", "investment banking"]

# Define the dataset file path
dataset_file = "dataset.csv"

# Create the dataset file and write the header
with open(dataset_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Video Title", "Video Description", "Class"])

# Iterate through each class
for class_name in classes:
    # Search for videos using the class as the query
    try:
        search_response = youtube.search().list(
            part="snippet",
            q=class_name,
            type="video",
            maxResults=100
        ).execute()

        # Iterate through the search results
        for search_result in search_response.get("items", []):
            video_id = search_result["id"]["videoId"]
            video_title = search_result["snippet"]["title"]
            video_description = search_result["snippet"]["description"]

            # Write the video details and class to the dataset file
            with open(dataset_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([video_title, video_description, class_name])

    except HttpError as e:
        print("An HTTP error occurred:")
        print(e)

print("Dataset created successfully!")
