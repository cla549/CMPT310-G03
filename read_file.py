from docx import Document
import nltk
from nltk.tokenize import word_tokenize as wt
import pdfplumber
import pytesseract
import os
from PIL import Image
import pandas as pd
import sys
import pathlib


def read_docx(file):
    doc = Document(file)
    all_words = []
    for paragraph in doc.paragraphs:
        all_words.extend(paragraph.text.split())
    return all_words


def read_pdf(file):
    pdf_text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()

            if not text:
                img = page.to_image(resolution=150).original
                text = pytesseract.image_to_string(img)
                
            if text:
                pdf_text += text + "\n"
    return pdf_text


def read_image(file):
    img = Image.open(file)
    text = pytesseract.image_to_string(img)
    return text


def main():
    file = sys.argv[1]
    output_directory = pathlib.Path(sys.argv[2])
    _, type_ = os.path.splitext(file)
    type_ = type_.lower()
    print(type_)
    if type_ == '.docx':
        words = read_docx(file)
    elif type_ == '.pdf':
        data = read_pdf(file)
        words = wt(data)
    elif type_ in ['.png', '.jpg', 'jpeg']:
        data = read_image(file)
        words = wt(data)
    else:
        print(f"Wrong file type.")
        return
    df = pd.DataFrame({"words" : words})
    file_count = sum(1 for x in output_directory.iterdir() if x.is_file())
    file_name_base = 'transformed_data.csv'
    temp_path = pathlib.Path(file_name_base)
    new_filename = f"{temp_path.stem}_{file_count + 1}{temp_path.suffix}"
    df.to_csv(output_directory / new_filename, index=False)
        

if __name__ == '__main__':
    main()