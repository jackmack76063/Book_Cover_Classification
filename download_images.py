# Name: Mackenzie Jackson
# Class: CS 435: Deep Learning
# Final Project: Image Download
# Description: This program will download images from the given image URL's in a CSV file.
# The purpose is to have the downloaded images, and filepath information, in order to train 
# a CNN model for image classification
# Date: 05/28/26
# Citations: Script adapted from git user biwana
# Source URL: https://github.com/uchidalab/book-dataset/blob/master/scripts/download_images.py
import os
import codecs
import pandas as pd
from argparse import ArgumentParser
from urllib import request
from urllib.error import HTTPError, URLError


#this section will let the program run with the format:
# python thisprogram.py <output_folder> <csv_file>
# Example: 'python download_images.py downloaded_images filtered_books.csv'

parser = ArgumentParser()

#first terminal argument after script name will be called 'output_dirpath'
#this is where the downloaded images will be saved
parser.add_argument(
    'output_dirpath',
    type=str,
    help="output directory path"
)

#second argument will be called 'csv_filepath'
#This is the csv file the script will read
parser.add_argument(
    'csv_filepath',
    type=str,
    help='csv filepath'
)

#reads command line and stores them
args = parser.parse_args()

#read in csv of prepared data
with codecs.open(args.csv_filepath, mode='r', 
                 encoding='utf-8', 
                 errors='ignore') as csv_file:
    bookData = pd.read_csv(csv_file)

#checks if main output folder exists or not
if not os.path.isdir(args.output_dirpath):
    os.makedirs(args.output_dirpath)
    
#confirm where images are being saved
print('[Download images into "{}"]'.format(args.output_dirpath))

#This function downloads each image
#It sorts each image into a folder by 'Category', and names it by 'Filename'
def download_image(i):
    filename = bookData.iloc[i]['Filename']
    category = bookData.iloc[i]['Category']
    #builds final filepath where images are saved
    inner_output_dirpath = os.path.join(args.output_dirpath, category)
    #checks whether category folder already exists
    if not os.path.isdir(inner_output_dirpath):
        os.mkdir(inner_output_dirpath)
    output_filepath = os.path.join(inner_output_dirpath, filename)

    url = bookData.iloc[i]['Image URL']
    #checks whether image file already exists
    if not os.path.isfile(output_filepath):
        try:
            downloaded_img = request.urlopen(url)
            image_file = open(output_filepath, mode='wb')
            image_file.write(downloaded_img.read())
            downloaded_img.close()
            image_file.close()
            
        #skips bad URL links without crashing program    
        except HTTPError as e:
            print("HTTP error for:", url, "Error:", e.code)
        
        except URLError as e:
            print("URL error for:", url, "Error:", e.reason)

#function call to download each image by row in the csv file
for i in range(len(bookData)):
    download_image(i)
    
#create a new column in dataset to hold filepath
#This will streamline loading images when later training the CNN model
#Ex: 'downloaded_images/Romance/233452.jpg'
filepaths = []
for i in range(len(bookData)):
    filename = bookData.iloc[i]["Filename"]
    category = bookData.iloc[i]["Category"]
    
    filepath = os.path.join(
        args.output_dirpath,
        category,
        filename
    )
    
    filepaths.append(filepath)
    
bookData["Filepath"] = filepaths

#Drop columns besides filepath and category
bookData = bookData[["Filepath", "Category"]]

#save modified dataset
bookData.to_csv(
    "book_dataset_final.csv",
    index=False
)