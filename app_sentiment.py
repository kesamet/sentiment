import json
import os
import requests
import tempfile
from io import StringIO

import numpy as np
import streamlit as st
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

from serve_http import predict

# COLOUR_MAP = {
#     "t0": "#F5B7B1",
#     "t1": "#AED6F1",
#     "t2": "#00FEFE",  # cyan
#     "t3": "#FFFF00",  # yellow
#     "t4": "#00FF00",  # green
# }
COLOUR_MAP = ["#F5B7B1", "#FADBD8", "FDEDEC", "#EBF5FB", "#D6EAF8", "#AED6F1"]


_highlighted = "<span style='background-color: {colour}'>{text}</span>"
    

def print_text_with_tags(texts, y_prob):
    print_str = []
    tags = (np.array(y_prob) * 6).astype(int)

    for text, tag in zip(texts, tags):
        print_str.append(_highlighted.format(colour=COLOUR_MAP[tag], text=text))

    return "\n\n".join(print_str)


def read_pdf(pdf_filepath):
    output_string = StringIO()
    with open(pdf_filepath, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    text = output_string.getvalue()
    return text


@st.cache
def pdf_to_text(uploaded_file):
    pdf_file = uploaded_file.read()
    fh, temp_filename = tempfile.mkstemp()
    try:
        with open(temp_filename, "wb") as f:
            f.write(pdf_file)
            f.flush()
            return read_pdf(f.name)
    finally:
        os.close(fh)
        os.remove(temp_filename)


@st.cache
def clean_text(raw_text):
    """Split raw text into sentences."""
    raw_texts = raw_text.split(".\n")

    output_texts = list()
    for t0 in raw_texts:
        t1 = t0.replace("\n\n", " ").replace("\n", " ").strip()
        output_texts.append(t1 + ".")
    return output_texts


def sentiment_analyzer():
    st.title("Article Sentiment Analysis Demo")

    # url = st.text_input("Input API URL.")
    # token = st.text_input("Input token.")

    uploaded_file = st.file_uploader("Upload a PDF.")
    if uploaded_file is not None:
        raw_text = pdf_to_text(uploaded_file)
        texts = clean_text(raw_text)

        data = {"texts": texts}
        # data = json.dumps(data)

        # headers = {"Content-Type": "application/json"}
        # if token != "":
        #     headers.update({"X-Bedrock-Api-Token": token})
        #
        # response = requests.post(url, headers=headers, data=data)

        # y_prob = response.json()["y_prob"]
        # tags = response.json()["tags"]

        y_prob = predict(data)

        output_text = print_text_with_tags(texts, y_prob)

        st.subheader("Mean Sentiment Score")
        st.write(f"**{np.mean(y_prob):.4f}**")

        st.subheader("Output")
        st.markdown(output_text, unsafe_allow_html=True)


if __name__ == "__main__":
    sentiment_analyzer()
