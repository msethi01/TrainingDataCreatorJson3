import fitz  # PyMuPDF
import re
import random
import csv
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Function to extract text from a PDF file using PyMuPDF
def extract_text_from_pdf(pdf_path):
    try:
        document = fitz.open(pdf_path)  # Open the PDF file
        text = ''
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            page_text = page.get_text("text")  # Extract text in "text" format
            if page_text.strip():  # Check if the page text is not empty
                print(f"Extracting text from page {page_num + 1}")
                text += page_text + '\n'
        document.close()
        print("Complete extracted text:")
        print(text[:1000])  # Print the first 1000 characters of the extracted text for verification
        return text
    except Exception as e:
        print("Error processing PDF:", e)
        return None

# Function to preprocess the extracted text
def preprocess_text(text):
    # Normalize case
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    print("Preprocessed Text:")
    print(text[:1000])  # Print the first 1000 characters of the preprocessed text
    return text

# Function to split the text data into training, validation, and test sets
def split_data(text, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Split text into sentences
    sentences = sent_tokenize(text)
    print("First 5 sentences after tokenization:")
    print(sentences[:5])  # Print the first 5 sentences for verification
    random.shuffle(sentences)
    total_sentences = len(sentences)
    print(f"Total sentences: {total_sentences}")

    train_end = int(total_sentences * train_ratio)
    val_end = train_end + int(total_sentences * val_ratio)

    train_data = sentences[:train_end]
    val_data = sentences[train_end:val_end]
    test_data = sentences[val_end:]

    print(f"Training sentences: {len(train_data)}")
    print(f"Validation sentences: {len(val_data)}")
    print(f"Test sentences: {len(test_data)}")

    return train_data, val_data, test_data

# Function to save the data to a CSV file
def save_data_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text'])
        for sentence in data:
            if sentence.strip():  # Check if the sentence is not empty
                writer.writerow([sentence.strip()])

# Main function to process the PDF and create datasets
def process_pdf_to_datasets(pdf_path):
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    if text is None:
        print("Failed to extract text from PDF.")
        return

    # Preprocess the text
    cleaned_text = preprocess_text(text)

    # Split the data into training, validation, and test sets
    train_data, val_data, test_data = split_data(cleaned_text)

    # Save the datasets to CSV files
    save_data_to_csv(train_data, 'train_data.csv')
    save_data_to_csv(val_data, 'val_data.csv')
    save_data_to_csv(test_data, 'test_data.csv')

    print("Datasets created successfully:")
    print("Training data: train_data.csv")
    print("Validation data: val_data.csv")
    print("Test data: test_data.csv")

# Example usage
pdf_path = 'fcug.pdf'  # Update with the path to your PDF file
process_pdf_to_datasets(pdf_path)
