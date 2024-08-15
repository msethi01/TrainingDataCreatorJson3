import fitz  # PyMuPDF
import re
import json
import random
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def clean_text(text):
    """Cleans the extracted text, removes specific unwanted lines and phrases."""
    text = re.sub(r'\u2022', '- ', text)  # Replaces bullet points with a hyphen and space
    text = re.sub(r'\u2122', '', text)  # Removes the trademark symbol
    text = re.sub(r'Fusion Compiler™ User Guide', '', text)
    text = re.sub(r'V-2023.12-SP3', '', text)
    text = re.sub(r'(?i)user\s?guide\s?\d+\s?chapter\s?\d+', '', text)  # Removes "User Guide <number> Chapter <number>"
    text = text.replace('\u00ae', '(R)')  # Registered trademark symbol
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Curly quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Curly single quotes
    text = text.replace('\u201a', '"').replace('\u201b', '"')
    text = re.sub(r'\b(Courier|italic|bold|purple|edit)\b', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text





def extract_text_with_headers(pdf_path):
    """Extracts text with formatting details and associates paragraphs with headers."""
    document = fitz.open(pdf_path)
    headers_with_paragraphs = []
    current_header = None
    current_paragraph = ""

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]  # Get text as dictionary blocks
        print(f"Extracting text from page {page_num + 1}")
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        cleaned_text = clean_text(span["text"])
                        if cleaned_text.strip():
                            # Detect headers based on font size and boldness
                            if span["size"] >= 10 and (span["flags"] & 2 > 0):  # Adjust these thresholds based on PDF
                                if current_header and current_paragraph.strip():
                                    headers_with_paragraphs.append({
                                        "header": current_header,
                                        "text": current_paragraph.strip()
                                    })
                                current_header = cleaned_text
                                current_paragraph = ""
                            else:
                                # Check if the text is part of a bullet or numbered list
                                if re.match(r'^\d+\.', cleaned_text) or re.match(r'^[-•]', cleaned_text):
                                    current_paragraph += "\n" + cleaned_text
                                else:
                                    current_paragraph += cleaned_text + " "

    # Save the last header and paragraph
    if current_header:
        headers_with_paragraphs.append({
            "header": current_header,
            "text": current_paragraph.strip()
        })

    document.close()
    return headers_with_paragraphs



def identify_headers(text_blocks, size_threshold=11, bold_threshold=True):
    """Identifies headers based on font size and boldness."""
    headers = []
    current_header = None
    current_content = []

    for block in text_blocks:
        if block["size"] >= size_threshold or block["bold"] == bold_threshold:
            if current_header:
                headers.append((current_header, ' '.join(current_content)))
            current_header = block["text"]
            current_content = []
        else:
            current_content.append(block["text"])

    if current_header:
        headers.append((current_header, ' '.join(current_content)))

    return headers



def remove_unwanted_lines(text, pattern):
    """Removes lines that match the pattern 'User Guide <number> Chapter <number>'."""
    lines = text.splitlines()
    filtered_lines = []

    for line in lines:
        if re.search(pattern, line, re.IGNORECASE):
            print(f"Removing line: {line}")
        else:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)





def find_start_of_instructions(text):
    """Finds the start of the relevant instructional content, triggered the 2nd time a keyword is encountered."""
    lower_text = text.lower()
    start_keywords = [
        "working with the fusion compiler tool",
        "physical synthesis design flow overview"
    ]
    start_idx = -1

    for keyword in start_keywords:
        first_idx = lower_text.find(keyword)  # Find the first occurrence
        if first_idx != -1:
            second_idx = lower_text.find(
                keyword,
                first_idx + len(keyword))  # Find the second occurrence
            if second_idx != -1:
                start_idx = second_idx
                break

    if start_idx == -1:
        start_idx = 1000  # Adjust this based on your document structure

    return text[start_idx:]





def split_into_paragraphs(text):
    """Splits the text into questions (headers) and their corresponding answers."""
    paragraphs = []
    current_header = None
    current_paragraph = []

    # Regex to detect headers (you may need to adjust this based on the specific format of headers in your PDF)
    header_pattern = re.compile(r'^[A-Z].*?:$')

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        if header_pattern.match(line):
            if current_header:
                # Save the previous header and its paragraph
                #paragraphs.append((current_header, ' '.join(current_paragraph)))
                paragraphs.append({
                    "question": current_header,
                    "answer": ' '.join(current_paragraph)
                })
            # Start a new paragraph
            current_header = line
            current_paragraph = []
        else:
            current_paragraph.append(line)

    # Don't forget to add the last paragraph
    if current_header:
        #paragraphs.append((current_header, ' '.join(current_paragraph)))
        paragraphs.append({
            "question": current_header,
            "answer": ' '.join(current_paragraph)
        })





def save_paragraphs_to_jsonl(paragraphs, output_file_path):
    """Saves the paragraphs to a JSONL file."""
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for header, content in paragraphs:
            json.dump({"question": header, "answer": content}, output_file)
            output_file.write('\n')

def save_headers_to_jsonl(headers_with_paragraphs, output_file_path):
    """Saves the headers and their content to a JSONL file."""
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for item in headers_with_paragraphs:
            output_file.write(json.dumps(item) + '\n')

def save_to_file(content, filename):
    """Utility function to save text content to a file."""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

def save_text_blocks_to_jsonl(text_blocks, output_file_path):
    """Saves the text blocks to a JSONL file."""
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for block in text_blocks:
            output_file.write(json.dumps(block) + '\n')



def save_text_blocks_to_text(text_blocks, output_text_file):
    """Saves the text blocks to a plain text file for debugging purposes."""
    with open(output_text_file, 'w', encoding='utf-8') as output_file:
        for block in text_blocks:
            output_file.write(f"Text: {block['text']}\n")
            output_file.write(f"Size: {block['size']}\n")
            output_file.write(f"Bold: {block['bold']}\n")
            output_file.write("\n" + "-"*80 + "\n\n")


def pdf_to_paragraphs_jsonl(pdf_file_path, output_jsonl_path):
    """Main function to extract text, identify headers, and save to JSONL."""
    text_blocks = extract_text_with_formatting(pdf_file_path)

    # Save the raw extracted text blocks for inspection
    save_text_blocks_to_jsonl(text_blocks, "text_blocks.jsonl")

    # Save text blocks to a plain text file
    save_text_blocks_to_text(text_blocks, "text_blocks.txt")
    
    # Combine the text blocks into a single text string for further processing
    combined_text = ' '.join([block['text'] for block in text_blocks])
    
    # Save the combined text before finding the start of instructions
    save_to_file(combined_text, "combined_text.txt")
    
    # Find the start of relevant instructions
    relevant_text = find_start_of_instructions(combined_text)
    
    # Save the relevant text
    save_to_file(relevant_text, "relevant_text.txt")
    
    # Further process relevant text if needed (e.g., additional cleaning)
    cleaned_relevant_text = clean_text(relevant_text)
    
    # Save the cleaned relevant text
    save_to_file(cleaned_relevant_text, "cleaned_relevant_text.txt")
    
    # Identify headers and their corresponding content
    headers = identify_headers(text_blocks)
    
    # Save headers and content to JSONL
    save_headers_to_jsonl(headers, output_jsonl_path)
    
    print(f"Headers and content have been saved to {output_jsonl_path}")

# Example usage
pdf_file_path = "fcug.pdf"  # Replace with your PDF file path
output_jsonl_path = "paragraphs.jsonl"

headers_with_paragraphs = extract_text_with_headers(pdf_file_path)
save_headers_to_jsonl(headers_with_paragraphs, "headers_with_paragraphs.jsonl")