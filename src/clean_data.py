# Name: Mackenzie Jackson
# Class: CS 435: Deep Learning
# Final Project: Data Preparation
# Description: This program will take in the CSV file (of books), filter categories, 
# then clean up unwanted columns, and save. 
# Date: 05/28/26

import pandas as pd

#read in CSV file and create headers
header_names = [
    "Amazon ID (ASIN)",
    "Filename",
    "Image URL",
    "Title",
    "Author",
    "Category ID",
    "Category"
]

bookData = pd.read_csv(
    "book32-listing.csv",
    encoding="latin1",
    header=None,
    names=header_names
)

#Filter out unwanted categories
keep_categories = [
    "Arts & Photography",
    "Biographies & Memoirs",
    "Children's Books",
    "Comics & Graphic Novels",
    "Cookbooks, Food & Wine",
    "Mystery, Thriller & Suspense",
    "Romance",
    "Science Fiction & Fantasy",
    "Travel",
    "Religion & Spirituality"
]

bookData = bookData[bookData["Category"].isin(keep_categories)]

#sample balanced counts from categories, giving 1500 images per caategorie
# Sample evenly from each category
bookData = (
    bookData
    .groupby("Category", group_keys=False)
    .sample(n=1500, random_state=42)
)

#Filter out unwanted columns from dataset
bookData = bookData[["Filename", "Image URL", "Category"]]

print(bookData["Category"].value_counts())

#Save prepared dataset
bookData.to_csv("filtered_books.csv", index=False)
