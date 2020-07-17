"""
Script for the app.
"""
import json
import os
import tempfile
from io import StringIO

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


def read_pdf(pdf_filepath, pages="all"):
    """Read PDF to output text."""
    output_string = StringIO()
    with open(pdf_filepath, "rb") as in_file:
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
def analyse(data, url, token):
    """Analyse."""
    if url != "" and token != "":
        import requests
        headers = {"Content-Type": "application/json"}
        if token != "":
            headers.update({"X-Bedrock-Api-Token": token})

        response = requests.post(url, headers=headers, data=json.dumps(data))
        y_prob = response.json()["y_prob"]
    else:
        from serve_http import predict
        y_prob = predict(data)
    return np.array(y_prob)


def plot_kde(scores):
    """Plot KDE as line chart."""
    colours = {
        "r": {
            "area_color": "#F1948A",
            "line_color": "#CB4335",
        },
        "b": {
            "area_color": "#85C1E9",
            "line_color": "#2E86C1",
        }
    }

    median_val = np.median(scores)
    c = "r" if median_val < 0 else "b"

    x, y = sns.kdeplot(scores, bw=0.05).get_lines()[0].get_data()
    source = pd.DataFrame({"x": x, "y": y})
    base = alt.Chart(source).mark_area(
        color=colours[c]["area_color"], fillOpacity=0.5
    ).encode(
        alt.X("x:Q", title="Sentiment score"), alt.Y("y:Q", title="KDE")
    )
    line = base.mark_line(color=colours[c]["line_color"])
    return base + line


def plot_ts(scores):
    """Plot time series as bar chart."""
    source = pd.DataFrame({"sentence_num": np.arange(len(scores)) + 1, "score": scores})
    bars = alt.Chart(source).mark_bar().encode(
        x=alt.X("sentence_num:Q", title="Sentence number"),
        y=alt.Y("score:Q", title="Sentiment score", scale=alt.Scale(domain=[-1, 1])),
        color=alt.condition(
            alt.datum.score > 0,
            alt.value("#2874A6"),  # The positive color
            alt.value("#B03A2E")  # The negative color
        )
    )
    return bars


def highlight_text_with_proba(sentences, y_prob):
    """Return highlighted text using probabilities."""
    colour_range = ["#F5B7B1", "#FADBD8", "#FDEDEC", "#EBF5FB", "#D6EAF8", "#AED6F1"]

    tags = (y_prob * 6).astype(int)
    _highlighted = "<span style='background-color: {colour}'>{text}</span>"
    print_str = []
    for sent, tag in zip(sentences, tags):
        print_str.append(_highlighted.format(colour=colour_range[tag], text=sent))
    return "\n\n".join(print_str)


def _present_results(sentences, y_prob):
    scores = y_prob * 2 - 1

    st.write(f"**Median Sentiment Score = `{np.median(scores):.4f}`**")
    # c1 = plot_kde(scores)
    # c2 = plot_ts(scores)
    # st.altair_chart(
    #     alt.concat(c1, c2, columns=2),
    #     use_container_width=True,
    # )
    st.altair_chart(plot_kde(scores), use_container_width=True)
    st.altair_chart(plot_ts(scores), use_container_width=True)

    st.subheader("Output")
    output_text = highlight_text_with_proba(sentences, y_prob)
    st.markdown(output_text, unsafe_allow_html=True)


def sentiment_analyzer():
    """Entry point of the app."""
    st.title("Article Sentiment Analysis Demo")

    url = st.text_input("Input API URL.")
    token = st.text_input("Input token.")

    uploaded_file = st.file_uploader("Upload a PDF.")
    page_nums = st.text_input("Input page numbers in the form of '2,3,4' or type 'all'.")
    if uploaded_file is not None and page_nums != "":
        pages = page_nums
        if page_nums != "all":
            pages = list(map(lambda x: int(x) - 1, page_nums.split(",")))

        raw_text = pdf_to_text(uploaded_file, pages=pages)
        sentences = clean_text(raw_text)

        data = {"sentences": sentences}
        y_prob = analyse(data, url, token)

        _present_results(sentences, y_prob)


if __name__ == "__main__":
    sentiment_analyzer()
