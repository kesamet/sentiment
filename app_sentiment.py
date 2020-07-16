"""
Script for the app.
"""
# import json
import os
import tempfile
from io import StringIO

# import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns
from nltk import tokenize
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

from serve_http import predict

COLOUR_MAP = ["#F5B7B1", "#FADBD8", "#FDEDEC", "#EBF5FB", "#D6EAF8", "#AED6F1"]
COLORS = {
    "r": {
        "area_color": "#F1948A",
        "line_color": "#CB4335",
    },
    "b": {
        "area_color": "#85C1E9",
        "line_color": "#2E86C1",
    }
}


def highlight_text_with_proba(sentences, y_prob):
    """Return highlighted text using probabilities."""
    tags = (np.array(y_prob) * 6).astype(int)

    _highlighted = "<span style='background-color: {colour}'>{text}</span>"
    print_str = []
    for sent, tag in zip(sentences, tags):
        print_str.append(_highlighted.format(colour=COLOUR_MAP[tag], text=sent))
    return "\n\n".join(print_str)


def read_pdf(pdf_filepath, pages="all"):
    """Read PDF to output text."""
    output_string = StringIO()
    with open(pdf_filepath, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for i, page in enumerate(PDFPage.create_pages(doc)):
            if pages != "all" and i not in pages:
                continue
            interpreter.process_page(page)

    text = output_string.getvalue()
    return text


@st.cache
def pdf_to_text(uploaded_file, pages):
    """Wrapper for read_pdf."""
    pdf_file = uploaded_file.read()
    fh, temp_filename = tempfile.mkstemp()
    try:
        with open(temp_filename, "wb") as f:
            f.write(pdf_file)
            f.flush()
            return read_pdf(f.name, pages)
    finally:
        os.close(fh)
        os.remove(temp_filename)


@st.cache
def clean_text(raw_text):
    """Split raw text into sentences."""
    raw_sents = tokenize.sent_tokenize(raw_text)

    sentences = list()
    for sent in raw_sents:
        sentences.append(
            sent.replace("\n\n", " ") .replace("\n", " ") .replace("\x00", "fi"))
    return sentences


@st.cache
def analyse(data, url=None, token=None):
    """Analyse."""
    # headers = {"Content-Type": "application/json"}
    # if token != "":
    #     headers.update({"X-Bedrock-Api-Token": token})
    #
    # response = requests.post(url, headers=headers, data=json.dumps(data))
    # y_prob = response.json()["y_prob"]

    y_prob = predict(data)
    return y_prob


def plot_kde(y_prob):
    """Plot KDE."""
    median_val = np.median(y_prob)
    c = "r" if median_val < 0.5 else "b"

    x, y = sns.kdeplot(y_prob, bw=0.05).get_lines()[0].get_data()
    source = pd.DataFrame({"x": x, "y": y})
    base = alt.Chart(source).mark_area(color=COLORS[c]["area_color"], fillOpacity=0.5).encode(
        alt.X("x:Q", title="Sentiment Score"), alt.Y("y:Q", title=""))
    line = base.mark_line(color=COLORS[c]["line_color"])
    return base + line


def sentiment_analyzer():
    """Entry point of the app."""
    st.title("Article Sentiment Analysis Demo")

    # url = st.text_input("Input API URL.")
    # token = st.text_input("Input token.")

    uploaded_file = st.file_uploader("Upload a PDF.")
    page_nums = st.text_input("Input page numbers in the form of '2,3,4' or type 'all'.")
    if uploaded_file is not None and page_nums != "":
        pages = page_nums
        if page_nums != "all":
            pages = list(map(lambda x: int(x) - 1, page_nums.split(",")))

        raw_text = pdf_to_text(uploaded_file, pages=pages)
        sentences = clean_text(raw_text)

        data = {"sentences": sentences}
        y_prob = analyse(data)

        _y_prob = y_prob[4: -2]  # to remove possible header and footer
        st.write(f"**Median Sentiment Score = {np.median(_y_prob):.4f}**")
        st.altair_chart(plot_kde(_y_prob), use_container_width=True)

        st.subheader("Output")
        output_text = highlight_text_with_proba(sentences, y_prob)
        st.markdown(output_text, unsafe_allow_html=True)


if __name__ == "__main__":
    sentiment_analyzer()
