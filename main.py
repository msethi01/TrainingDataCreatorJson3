import fitz  # PyMuPDF
import re
import json
import random
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

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
        return text
    except Exception as e:
        print("Error processing PDF:", e)
        return None

def clean_text(text):
    """Cleans the extracted text."""
    # Replace bullet characters with a placeholder or remove them
    text = text.replace('\u2022', '- ')  # Replaces bullet points with a hyphen and space
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def find_start_of_instructions(text):
    """Finds the start of the relevant instructional content, triggered the 2nd time a keyword is encountered."""
    lower_text = text.lower()
    start_keywords = ["working with the fusion compiler tool", "physical synthesis design flow overview"]
    start_idx = -1

    for keyword in start_keywords:
        first_idx = lower_text.find(keyword)  # Find the first occurrence
        if first_idx != -1:
            second_idx = lower_text.find(keyword, first_idx + len(keyword))  # Find the second occurrence
            if second_idx != -1:
                start_idx = second_idx
                break

    if start_idx == -1:
        start_idx = 1000  # Adjust this based on your document structure

    return text[start_idx:]

def split_into_paragraphs(text):
    """Splits the text into paragraphs to make it easier to process."""
    # Split on either double newlines, single newlines, or periods followed by a space
    paragraphs = re.split(r'\.\s|\n\s*\n|\n', text)
    for i, para in enumerate(paragraphs):
        print(f"Paragraph {i+1}: {para[:100]}...")  # Print the first 50 characters of each paragraph
    return [p.strip() for p in paragraphs if p.strip()]

def extract_instructions(text):
    """Extracts potential instructional content from the text."""
    instructions = []
    paragraphs = split_into_paragraphs(text)

    for paragraph in paragraphs:
        # Expanded regex to capture more instructional patterns, including lists and bullet points
        matches = re.findall(r'(?i)(step \d+[:]?.*?|instruction.*?|note[:]?.*?|procedure.*?|guideline.*?|•.*?|[-*] .*?|\d+\..*?|\d+\)\s+.*?)(?=\n|$)', paragraph)
        if matches:
            print(f"Matches in paragraph: {[m[:50] for m in matches[:5]]}...")
        instructions.extend(matches)

    return instructions

def create_conversational_data(instructions):
    """Formats the instructions into a question-answer conversational style dataset."""
    data = []
    for instruction in instructions:
        question = f"How do I {instruction.split(':')[0].lower()}?"
        answer = instruction
        data.append({"instruction": question, "response": answer})
    return data

def split_data(data, train_size=0.8, val_size=0.1, test_size=0.1):
    if len(data) < 2:
        return data, [], []
    elif len(data) < 3:
        train_data, test_data = train_test_split(data, test_size=0.5, random_state=42)
        return train_data, [], test_data
    else:
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=val_size/(train_size+val_size), random_state=42)
        return train_data, val_data, test_data

def save_to_jsonl(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        for item in data:
            output_file.write(json.dumps(item) + '\n')

def pdf_to_conversational_data(pdf_file_path, output_dir, train_size=0.8, val_size=0.1, test_size=0.1):
    print(f"Extracting text from {pdf_file_path}...")
    text = extract_text_from_pdf(pdf_file_path)

    if text is None:
        print("No text extracted from the PDF.")
        return

    print("Cleaning extracted text...")
    cleaned_text = clean_text(text)

    print("Finding start of relevant instructions...")
    relevant_text = find_start_of_instructions(cleaned_text)
    print("Relevant text starts with:")
    print(relevant_text[:100])  # Print the first 100 characters

    print("Extracting instructions from the text...")
    instructions = extract_instructions(relevant_text)
    print(f"Number of instructions extracted: {len(instructions)}")

    if not instructions:
        print("No instructions found in the text.")
        return

    print("Creating conversational data...")
    conversational_data = create_conversational_data(instructions)

    if len(conversational_data) < 3:
        print("Not enough data to split into train, val, and test. Using all data for training.")
        train_data, val_data, test_data = conversational_data, [], []
    else:
        print("Splitting data into training, validation, and test sets...")
        train_data, val_data, test_data = split_data(conversational_data, train_size, val_size, test_size)

    print(f"Saving training data to {output_dir}/train.jsonl...")
    save_to_jsonl(train_data, f"{output_dir}/train.jsonl")

    if val_data:
        print(f"Saving validation data to {output_dir}/val.jsonl...")
        save_to_jsonl(val_data, f"{output_dir}/val.jsonl")

    if test_data:
        print(f"Saving test data to {output_dir}/test.jsonl...")
        save_to_jsonl(test_data, f"{output_dir}/test.jsonl")

    print("Conversion complete!")

# Example usage
pdf_file_path = "fcug.pdf"  # Replace with your PDF file path
output_dir = "."  # Replace with your desired output directory
pdf_to_conversational_data(pdf_file_path, output_dir)
