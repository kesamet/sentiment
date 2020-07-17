"""
Script for the app.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns


@st.cache
def read_data(file_path):
    return json.load(open(file_path, "r"))


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
        x=alt.X("sentence_num:N", title="Sentence number"),
        y=alt.Y("score:Q", title="Sentiment score", scale=alt.Scale(domain=[-1, 1])),
        color=alt.condition(
            alt.datum.score > 0,
            alt.value("#2874A6"),  # The positive color
            alt.value("#B03A2E"),  # The negative color
        ),
        tooltip=["sentence_num", "score"],
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
    st.write("The KDE plot shows the distribution of the sentiment scores of all the sentences. "
             "A left-skewed plot indicates generally positive sentiment for the entire article.")
    st.altair_chart(plot_kde(scores), use_container_width=True)
    st.write("The plot below shows the sequence of sentiment scores in successive order.")
    st.altair_chart(plot_ts(scores), use_container_width=True)

    sent_idx = st.slider("Select sentence.", 1, len(y_prob))
    output_sent = highlight_text_with_proba(
        sentences[sent_idx - 1: sent_idx], y_prob[sent_idx - 1: sent_idx])
    st.markdown(output_sent, unsafe_allow_html=True)

    st.subheader("Full Text Output")
    output_text = highlight_text_with_proba(sentences, y_prob)
    st.markdown(output_text, unsafe_allow_html=True)


def demo_sentiment_analyzer():
    st.title("Article Sentiment Analysis Demo")
    st.subheader("Analysis Method")
    st.write("- Given an article in PDF or text, we clean and split the text into sentences.\n"
             "- Each sentence is then fed into a ML model to generate a sentiment score.\n"
             "- Sentiment score ranges between `-1.0` (most negative) and `1.0` (most positive).\n")

    st.subheader("Examples (in descending order of median sentiment score)")
    select_ex = st.selectbox("Select examples.", [f"ex{i}" for i in range(1, 5)])
    data = read_data(f"results/{select_ex}.txt")
    st.write("Article is from {}.".format(data["url"]))

    _present_results(data["sentences"], np.array(data["y_prob"]))


if __name__ == "__main__":
    demo_sentiment_analyzer()
