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
    
def clean_text(text):
    """Cleans the extracted text, removes specific unwanted lines and phrases."""
    # Remove lines like 'User Guide 175 Chapter 2'
    text = remove_unwanted_lines(text, r'(?i)user\s?guide\s?\d+\s?chapter\s?\d+')

    # Replace bullet characters with a placeholder or remove them
    text = text.replace('\u2022', '- ')  # Replaces bullet points with a hyphen and space
    text = text.replace('\u2122', '')  # Removes the trademark symbol
    text = text.replace('Fusion Compiler™ User Guide', '')
    text = text.replace('V-2023.12-SP3', '')

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Debug: Print the first 500 characters of cleaned text
    print("Cleaned text starts with:")
    print(text[:500])

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

#def split_into_paragraphs(text):
#    """Splits the text into paragraphs to make it easier to process."""
#    # Split on either double newlines, single newlines, or periods followed by a space
#    paragraphs = re.split(r'\.\s|\n\s*\n|\n', text)
#    for i, para in enumerate(paragraphs):
#        print(f"Paragraph {i+1}: {para[:100]}...")  # Print the first 50 characters of each paragraph
#    return [p.strip() for p in paragraphs if p.strip()]


#def split_into_paragraphs(text):
#    """Splits the text into paragraphs based on double newlines or significant line breaks."""
#    # Split on double newlines (common paragraph separator) or a single newline followed by an indentation
#    paragraphs = re.split(r'\n\s*\n+', text)
#
#    # Further clean up each paragraph to remove leading/trailing spaces and merge lines broken by single newlines
#    paragraphs = [re.sub(r'\n+', ' ', para).strip() for para in paragraphs]
#
#    # Print out the first few paragraphs to verify correctness
#    for i, para in enumerate(paragraphs[:5]):
#        print(f"Paragraph {i+1}: {para[:100]}...")  # Print the first 100 characters of each paragraph
#
#    return [p for p in paragraphs if p.strip()]

#def split_into_paragraphs(text):
#    """Splits the text into paragraphs using heuristics based on line length."""
#    lines = text.split('\n')
#    paragraphs = []
#    current_paragraph = []

#    for line in lines:
#        if len(line.strip()) < 50:  # Adjust this threshold based on your text
#            if current_paragraph:
#                paragraphs.append(' '.join(current_paragraph))
#                current_paragraph = []
#            paragraphs.append(line.strip())
#        else:
#            current_paragraph.append(line.strip())
#
#    if current_paragraph:
#        paragraphs.append(' '.join(current_paragraph))
#
#    # Print the first few paragraphs to verify
#    for i, para in enumerate(paragraphs[:5]):
#        print(f"Paragraph {i+1}: {para[:100]}...")
#
#    return paragraphs

def split_into_paragraphs(text):
    """Splits the text into paragraphs while preserving bullet points and avoiding splitting on periods."""

    # First, handle bullet points and lists, keeping them in the same paragraph
    paragraphs = re.split(r'\n{2,}|\n(?=\d+\.|\u2022|-)', text)  # Split on double newlines or lines starting with a bullet or number

    # Merge single newlines within paragraphs (because your text sample seems to use single newlines within paragraphs)
    paragraphs = [para.replace('\n', ' ').strip() for para in paragraphs]

    # Now, further split where necessary based on specific patterns, e.g., chapter or section breaks
    refined_paragraphs = []
    for para in paragraphs:
        # Look for patterns that should signify the start of a new paragraph
        # Example: splitting on "Fusion Compiler User Guide"
        sub_paragraphs = re.split(r'Fusion Compiler User Guide', para)
        refined_paragraphs.extend(sub_paragraphs)

    # Final cleanup
    refined_paragraphs = [p.strip() for p in refined_paragraphs if p.strip()]

    # Debug: Print the first few paragraphs to verify
    for i, para in enumerate(refined_paragraphs[:5]):
        print(f"Paragraph {i+1}: {para[:100]}...")

    return refined_paragraphs
        
#def split_into_paragraphs(text):
#    """Splits the text into paragraphs based on double newlines or a single newline followed by indentation."""
#    # Split on double newlines (common paragraph separator)
#    paragraphs = re.split(r'\n\s*\n+', text)

#    # Also consider a single newline followed by indentation (optional, depending on your text structure)
#    paragraphs = [re.sub(r'\n\s+', ' ', para).strip() for para in paragraphs]

#    for i, para in enumerate(paragraphs):
#        print(f"Paragraph {i+1}: {para[:100]}...")  # Print the first 100 characters of each paragraph
#    return [p for p in paragraphs if p.strip()]

#def split_into_paragraphs(text):
#    """Splits the text into paragraphs more reliably by looking for double newlines or significant gaps."""
#    # This pattern splits paragraphs based on:
#    # - Double newlines (common paragraph separator)
#    # - Single newline followed by a capital letter (which often indicates a new sentence in a paragraph)
#    paragraphs = re.split(r'\n{2,}|\n(?=[A-Z])', text)
#
#    # Further clean up each paragraph to remove leading/trailing spaces and merge lines broken by single newlines
#    paragraphs = [re.sub(r'\n+', ' ', para).strip() for para in paragraphs]
#
#    for i, para in enumerate(paragraphs):
#        print(f"Paragraph {i+1}: {para[:100]}...")  # Print the first 100 characters of each paragraph
#    return [p for p in paragraphs if p.strip()]


#def split_into_paragraphs(text):
#    """Splits the text into paragraphs based on double newlines or indentation."""
#    # First, try splitting by double newlines, which is the most common way paragraphs are separated
#    paragraphs = text.split('\n\n')

#    # If that doesn't result in a good split, try to refine it
#    refined_paragraphs = []
#    for paragraph in paragraphs:
#        # If a "paragraph" is too long, it may contain multiple actual paragraphs split by single newlines
#        sub_paragraphs = paragraph.split('\n')

#        # Combine lines that are not actually the start of a new paragraph
#        combined_paragraph = []
#        for sub_para in sub_paragraphs:
#            if combined_paragraph and (sub_para and sub_para[0].isupper()):
#                # If the current line starts with an uppercase letter, treat it as the start of a new paragraph
#                refined_paragraphs.append(' '.join(combined_paragraph))
#                combined_paragraph = [sub_para]
#            else:
#                combined_paragraph.append(sub_para)

#        # Don't forget to add the last combined paragraph
#        if combined_paragraph:
#            refined_paragraphs.append(' '.join(combined_paragraph))

#    # Clean up each paragraph to remove leading/trailing spaces
#    refined_paragraphs = [p.strip() for p in refined_paragraphs if p.strip()]

#    # Print the first few paragraphs for debugging
#    for i, para in enumerate(refined_paragraphs[:5]):
#        print(f"Paragraph {i+1}: {para[:100]}...")  # Print the first 100 characters of each paragraph
#
#    return refined_paragraphs

#def split_into_paragraphs(text):
#    """Splits the text into paragraphs based on more refined conditions."""
#    # Attempt to split paragraphs based on a combination of double newlines and keeping single lines intact
#    paragraphs = re.split(r'(?<=\n)\s*(?=\n)|(?<=\.)\s*\n(?=[A-Z])', text)
#
#    # Clean up each paragraph to remove leading/trailing spaces and rejoin single lines that were split unnecessarily
#    paragraphs = [re.sub(r'\s*\n\s*', ' ', para).strip() for para in paragraphs if para.strip()]
#
#    # Print out the first few paragraphs to verify correctness
#    for i, para in enumerate(paragraphs[:5]):
#        print(f"Paragraph {i+1}: {para[:100]}...")  # Print the first 100 characters of each paragraph
#
#    return paragraphs


#def extract_instructions(text):
#    """Extracts potential instructional content from the text."""
#    instructions = []
#    paragraphs = split_into_paragraphs(text)
#
#    for paragraph in paragraphs:
#
#        # Skip paragraphs that contain unwanted phrases
#        if 'Fusion Compiler™ User Guide' in paragraph or 'V-2023.12-SP3' in paragraph:
#            continue
#            
#        # Expanded regex to capture more instructional patterns, including lists and bullet points
#        matches = re.findall(r'(?i)(step \d+[:]?.*?|instruction.*?|note[:]?.*?|procedure.*?|guideline.*?|•.*?|[-*] .*?|\d+\..*?|\d+\)\s+.*?)(?=\n|$)', paragraph)
#        if matches:
#            print(f"Matches in paragraph: {[m[:50] for m in matches[:5]]}...")
#        instructions.extend(matches)
#
#    return instructions


#def extract_instructions(text):
#    """Extracts potential instructional content from the text."""
#    instructions = []
#    paragraphs = split_into_paragraphs(text)
#
#    for paragraph in paragraphs:
#        # Skip paragraphs that contain unwanted phrases
#        if 'Fusion Compiler™ User Guide' in paragraph or 'V-2023.12-SP3' in paragraph:
#            continue
#
#        # Expanded regex to capture more instructional patterns, including lists and bullet points
#        matches = re.findall(r'(?i)(step \d+[:]?.*?|instruction.*?|note[:]?.*?|procedure.*?|guideline.*?|•.*?|[-*] .*?|\d+\..*?|\d+\)\s+.*?)(?=\n|$)', paragraph)
#        if matches:
#            print(f"Matches in paragraph: {[m[:50] for m in matches[:5]]}...")
#        instructions.extend(matches)
#
#    return instructions

def extract_instructions(text):
    """Extracts potential instructional content from the text."""
    instructions = []
    paragraphs = split_into_paragraphs(text)

    for paragraph in paragraphs:
        # Skip paragraphs that contain unwanted phrases
        if 'Fusion Compiler™ User Guide' in paragraph or 'V-2023.12-SP3' in paragraph:
            continue

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

    # Write the cleaned text to a file for analysis
    cleaned_text_file_path = f"{output_dir}/cleaned_text.txt"
    with open(cleaned_text_file_path, 'w', encoding='utf-8') as cleaned_text_file:
        cleaned_text_file.write(cleaned_text)

    
    print("Finding start of relevant instructions...")
    relevant_text = find_start_of_instructions(cleaned_text)
    print("Relevant text starts with:")
    print(relevant_text[:100])  # Print the first 100 characters

    relevant_text_file_path = f"{output_dir}/relevant_text.txt"
    with open(relevant_text_file_path, 'w', encoding='utf-8') as relevant_text_file:
        relevant_text_file.write(relevant_text)
    
    print("Extracting instructions from the text...")
    instructions = extract_instructions(relevant_text)
    print(f"Number of instructions extracted: {len(instructions)}")

    instructions_file_path = f"{output_dir}/instructions.txt"
    with open(instructions_file_path, 'w', encoding='utf-8') as instructions_file:
        for instruction in instructions:
            instructions_file.write(instruction + '\n')
    print(f"Extracted instructions have been written to {instructions_file_path}.")

        
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
