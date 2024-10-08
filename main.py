import fitz  # PyMuPDF
import re
import json
import random
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import csv 

nltk.download('punkt')

# Initialize the tokenizer (use the appropriate model name for your specific model)
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace with your model's tokenizer

def clean_text(text):
    """Cleans the extracted text, removes specific unwanted lines and phrases."""
    text = re.sub(r'\u2022', '- ', text)  # Replaces bullet points with a hyphen and space
    text = re.sub(r'\u2122', '', text)  # Removes the trademark symbol
    #text = re.sub(r'Fusion Compiler™ User Guide', '', text)
    #text = re.sub(r'Fusion Compiler User Guide', '', text)
    text = re.sub(r'^Chapter.*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Figure.*\n?', '', text)
    text = re.sub(r'V-2023.12-SP3', '', text)
    text = re.sub(r'(?i)user\s?guide\s?\d+\s?chapter\s?\d+', '', text)  # Removes "User Guide <number> Chapter <number>"
    text = text.replace('\u00ae', '(R)')  # Registered trademark symbol
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Curly quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Curly single quotes
    text = text.replace('\u201a', '"').replace('\u201b', '"')
    text = text.replace('\u25ba', '')
    text = text.replace("(FILE-007)", "")
    text = re.sub(r'\b(Courier|italic|bold|purple|edit)\b', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-{2,}', '', text)

    # Remove any remaining unwanted Unicode characters
    text = re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)  # Removes \uXXXX Unicode sequences


    # Remove trailing hyphens or other placeholders if the text is empty
    if text == '-' or len(text) <= 1:
        text = ''
        
    return text


def clean_multiple_spaces(text):
    """Removes multiple spaces and replaces them with a single space."""
    return re.sub(r'\s+', ' ', text)


def remove_ing_from_first_word(header_text):
    """Removes 'ing' from the first word in a header if it ends with 'ing'."""
    words = header_text.split()
    if words and words[0].lower().endswith("ing"):
        # Remove 'ing' from the first word
        words[0] = words[0][:-3]  # Cut the 'ing' from the end
    return " ".join(words)
    
def save_headers_to_text(headers_with_paragraphs, output_text_file):
    """Saves headers and their paragraphs to a plain text file."""
    with open(output_text_file, 'w', encoding='utf-8') as output_file:
        for item in headers_with_paragraphs:
            header = item.get('header', 'No Header')
            text = item.get('text', 'No Text')
            size = item.get('size', 'Unknown Size')
            font = item.get('font', 'Unknown Font')
            flags = item.get('flags', 'Unknown Flags')

            output_file.write(f"Header: {header}\n")
            output_file.write(f"Size: {size}\n")
            output_file.write(f"Font: {font}\n")
            output_file.write(f"Flags: {flags}\n")
            output_file.write(f"Text:\n{text}\n")
            output_file.write("-" * 80 + "\n")

            
def extract_text_with_headers(pdf_path, clean, newline_after_paragraphs=False, debug_file_path="debug_log.txt"):
    """Extracts text with formatting details and associates paragraphs with headers and logs debug information."""
    document = fitz.open(pdf_path)
    headers_with_paragraphs = []
    current_header = None
    current_paragraph = ""
    start_of_instructions_found = False

    sentence_endings = re.compile(r'(?<=[.!?])\s')  # Regex to detect end of sentence followed by space

    # Open debug file for logging
    with open(debug_file_path, 'w', encoding='utf-8') as debug_file:

        for page_num in range(len(document)):
            page = document.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]  # Get text as dictionary blocks

            debug_file.write(f"Extracting text from page {page_num + 1}\n")
            print(f"Extracting text from page {page_num + 1}")

            for block in blocks:
                if "lines" in block:
                    # Pass all spans in the block to find_start_of_instructions
                    spans = [span for line in block["lines"] for span in line["spans"]]

                    if not start_of_instructions_found:
                        start_of_instructions_found = find_start_of_instructions(spans)

                    if start_of_instructions_found:
                        for line in block["lines"]:
                            line_text = ""  # Collect the entire line's text
                            is_header = False  # Track if any span in the line qualifies as a header

                            debug_file.write(f"\nProcessing line:\n")

                            for span in line["spans"]:
                                # Skip text with specific size, font, and flags
                                if ((span["size"] == 10.0 and span["font"] == "ArialMT" and span["flags"] == 4) or
                                    (span["size"] == 10.0 and span["font"] == "Helvetica-Bold" and span["flags"] == 20)):
                                    continue  # Skip this span and proceed to the next

                                text = span["text"]

                                # Conditionally clean the text
                                if clean:
                                    text = clean_text(text)

                                debug_file.write(f"Span text: {text}, Size: {span['size']}, Font: {span['font']}, Flags: {span['flags']}\n")

                                if text.strip():
                                    is_italic = "Arial-ItalicMT" in span["font"] and (span["size"] >= 11) and (span["flags"] == 6)

                                    # Check if this span qualifies as part of a header
                                    if ((span["size"] >= 11 and (span["flags"] & 2 > 0)) or
                                        (span["size"] >= 14 and span["font"] == "Arial-BoldMT" and span["flags"] == 20) or
                                        (span["size"] >= 20 and (span["flags"] & 20 > 0))) and not is_italic:
                                        is_header = True

                                    # Append the current span to the full line text
                                    line_text += text + " "

                            # If a header is detected for the entire line
                            if is_header:
                                debug_file.write(f"Header detected: {line_text.strip()}\n")
                                #current_header = remove_ing_from_first_word(line_text.strip())
                                current_header = line_text.strip()
                                
                                if current_header and current_paragraph.strip():
                                    cleaned_paragraph = clean_multiple_spaces(current_paragraph.strip())
                                    paragraphs = split_paragraph(cleaned_paragraph.strip(), 2048, current_header)
                                    for para in paragraphs:
                                        headers_with_paragraphs.append({
                                            "header": "How do I " + current_header,
                                            "size": span['size'],
                                            "font": span['font'],
                                            "flags": span['flags'],
                                            "text": para + ("\n" if newline_after_paragraphs else "")
                                        })
                                current_paragraph = ""
                            else:
                                # Append text to the current paragraph
                                current_paragraph += sentence_endings.sub('\n ', line_text.strip() + " ")

        # Add remaining paragraph text under the current header
        if current_header:
            cleaned_paragraph = clean_multiple_spaces(current_paragraph.strip())
            paragraphs = split_paragraph(cleaned_paragraph, 2048, current_header)
            for para in paragraphs:
                headers_with_paragraphs.append({
                    "header": "How do I " + current_header,
                    "text": para + ("\n" if newline_after_paragraphs else "")
                })

        document.close()

    print(f"Debug information has been logged to {debug_file_path}")
    return headers_with_paragraphs






    
def split_paragraph(paragraph, max_tokens, header):
    """Splits a paragraph into sections of no more than max_length tokens."""

    # Tokenize the paragraph
    tokens = tokenizer.tokenize(paragraph)

    # Calculate the number of tokens
    token_count = len(tokens)

    # If the paragraph is longer than the max tokens allowed
    if token_count > max_tokens:
        print(f"Splitting paragraph under header: {header} (Length: {token_count} tokens)")

    # Break the tokens into sections of max_tokens length
    sections = []
    for i in range(0, token_count, max_tokens):
        section_tokens = tokens[i:i + max_tokens]
        # Decode the tokens back to a string and add to the sections list
        sections.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(section_tokens)).strip())

    return sections

    
def transform_data_for_finetuning(data):
    """Transforms the dataset to the format required for fine-tuning LLaMA."""
    transformed_data = []
    for item in data:
        transformed_item = {
            "instruction": item["header"],  # Using 'header' as the question or instruction
            "input": "",  # No additional input context
            "output": item["text"]  # Using 'text' as the response
        }
        transformed_data.append(transformed_item)
    return transformed_data
    
def examine_text_properties(pdf_path, output_file_path):
    """Examines the properties of text spans in the PDF and logs any differences in font, text, size, or flags within the same line."""
    document = fitz.open(pdf_path)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]  # Get text as dictionary blocks
            output_file.write(f"Examining text properties on page {page_num + 1}\n")
            output_file.write("=" * 80 + "\n")

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        previous_span = None  # Keep track of the previous span

                        for span in line["spans"]:
                            current_font = span["font"]
                            current_text = span["text"]
                            current_size = span["size"]
                            current_flags = span["flags"]

                            # Write out the current span's properties to the file
                            output_file.write(f"Text: {current_text}\n")
                            output_file.write(f"Size: {current_size}\n")
                            output_file.write(f"Font: {current_font}\n")
                            output_file.write(f"Flags: {current_flags}\n")
                            output_file.write("-" * 40 + "\n")

                            # If there is a previous span, compare it with the current span
                            if previous_span:
                                prev_font = previous_span["font"]
                                prev_text = previous_span["text"]
                                prev_size = previous_span["size"]
                                prev_flags = previous_span["flags"]

                                # Check for differences in font, text, size, or flags
                                if current_font != prev_font or current_size != prev_size or current_flags != prev_flags:
                                    output_file.write("Difference detected within the line:\n")
                                    output_file.write(f"Previous -> Text: {prev_text}, Size: {prev_size}, Font: {prev_font}, Flags: {prev_flags}\n")
                                    output_file.write(f"Current  -> Text: {current_text}, Size: {current_size}, Font: {current_font}, Flags: {current_flags}\n")
                                    output_file.write("-" * 40 + "\n")

                            # Update previous_span to the current span for comparison
                            previous_span = span

    document.close()
    print(f"Text properties have been logged to {output_file_path}")


def find_start_of_instructions(spans):
    """Finds the start of the relevant instructional content based on specific font size, font, and keyword."""
    start_keywords = "physical synthesis design flow overview"

    for span in spans:
        cleaned_text = clean_text(span["text"]).lower()
        if (span["size"] == 11 and span["font"] == "ArialMT" and start_keywords in cleaned_text):
            return True  # Return True as soon as the conditions are met

    return False  # Return False if the conditions are not met

def transform_data_for_openai_finetuning(data):
    """Transforms the dataset to the format required for fine-tuning OpenAI's models."""
    transformed_data = []
    for item in data:
        transformed_item = {
            "messages": [
                {"role": "system", "content": "You are a chip design assistant. You should help the user to answer their question."},
                {"role": "user", "content": item["header"]},
                {"role": "assistant", "content": item["text"]}
            ]
        }
        transformed_data.append(transformed_item)
    return transformed_data


def save_headers_to_jsonl(headers_with_paragraphs, output_file_path):
    """Saves the headers and their content to a JSONL file."""
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for item in headers_with_paragraphs:
            output_file.write(json.dumps(item) + '\n')

def save_to_file(content, filename):
    """Utility function to save text content to a file."""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)



# Load the data from the headers_with_paragraphs.jsonl file
def load_jsonl_data(jsonl_file_path):
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def save_jsonl_data(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

# Split the data into training, validation, and testing sets
def split_data(data, train_size=0.8, val_size=0.1, test_size=0.1):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size / (train_size + val_size), random_state=42)
    return train_data, val_data, test_data


def split_openai_data(data, train_size=0.8, val_size=0.2):
    """Splits the OpenAI-formatted data into training and validation sets."""
    train_data, val_data = train_test_split(data, test_size=val_size, random_state=42)
    return train_data, val_data

def save_headers_to_csv_instruction_format(headers_with_paragraphs, output_csv_file):
    """Saves headers and paragraphs to a CSV file in the instruction, input, output format."""
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header row
        csv_writer.writerow(["Instruction", "Input", "Output"])
    
        # Write each entry as instruction, input, output
        for item in headers_with_paragraphs:
            instruction = item.get('header', 'No Header')
            output = item.get('text', 'No Text')
    
            # Write the row with instruction, input, and output. Input is empty in this case
            csv_writer.writerow([instruction, "", output])
    
            
# Example usage
pdf_file_path = "fcug.pdf"  # Replace with your PDF file path
output_jsonl_path = "paragraphs.jsonl"

examine_text_properties(pdf_file_path, "text_properties_debug.txt")

headers_with_paragraphs = extract_text_with_headers(pdf_file_path, False)
save_headers_to_jsonl(headers_with_paragraphs, "headers_with_paragraphs_noclean.jsonl")
save_headers_to_csv_instruction_format(headers_with_paragraphs, "headers_with_paragraphs_noclean_instruction_format.csv")

headers_with_paragraphs = extract_text_with_headers(pdf_file_path, True)
save_headers_to_jsonl(headers_with_paragraphs, "headers_with_paragraphs.jsonl")
save_headers_to_csv_instruction_format(headers_with_paragraphs, "headers_with_paragraphs_instruction_format.csv")

# Save to text format
save_headers_to_text(headers_with_paragraphs, "headers_with_paragraphs_noclean.txt")


jsonl_file_path = "headers_with_paragraphs.jsonl"
data = load_jsonl_data(jsonl_file_path)

# Transform the data for fine-tuning
transformed_data = transform_data_for_finetuning(data)
save_jsonl_data(transformed_data, "full_data.jsonl")

# Transform the data for fine-tuning OpenAI models
openai_transformed_data = transform_data_for_openai_finetuning(data)
save_jsonl_data(openai_transformed_data, "openai_full_data.jsonl")

# Split the OpenAI data into training and validation sets
openai_train_data, openai_val_data = split_openai_data(openai_transformed_data)

# Save the split OpenAI data to JSONL files
save_jsonl_data(openai_train_data, "openai_train_data.jsonl")
save_jsonl_data(openai_val_data, "openai_val_data.jsonl")


train_data, val_data, test_data = split_data(transformed_data)

# Save the split data to JSONL files
save_jsonl_data(train_data, "train_data.jsonl")
save_jsonl_data(val_data, "val_data.jsonl")
save_jsonl_data(test_data, "test_data.jsonl")

print("Training, validation, and testing JSONL files have been created.")

