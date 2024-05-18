from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import streamlit as st
import requests
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import html
import json
import warnings
warnings.filterwarnings("ignore")

# Get the current directory of app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to reach the project root
project_dir = os.path.join(current_dir, "..")
os.chdir(project_dir)
# @st.cache_data


def setup():
    reviewsDf = pd.read_csv(f"{project_dir}/data/ReviewsDataset.csv")

    labels = pd.read_csv(
        f"{project_dir}/data/LookupData/Labels.csv", index_col=0)
    reactions = pd.read_csv(
        f"{project_dir}/data/LookupData/Reactions.csv", index_col=0)
    events = pd.read_csv(
        f"{project_dir}/data/LookupData/Events.csv", index_col=0)

    with open(f"{project_dir}/data/LookupData/DrugMapping.json") as file:
        mapping = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
    model = AutoModelForTokenClassification.from_pretrained(
        "Clinical-AI-Apollo/Medical-NER")
    nlp_token_class = pipeline(
        'ner', model=model, tokenizer=tokenizer, grouped_entities=True)

    return {
        "reviews": reviewsDf,
        "labels": labels,
        "reactions": reactions,
        "events": events,
        "mapping": mapping,
        "tokens": nlp_token_class}


def getDrugs(response):
    pattern = r'Medicine\s*\"([^\"]+)\"'
    drugs = re.findall(pattern, response["answer"])
    drugs = list(set([drug.split(" ")[0].lower() for drug in drugs]))
    return drugs


def getData(drugs, reviewsDf, labels, reactions, events, mapping):

    REVIEWS = reviewsDf[reviewsDf.drug.isin(drugs)]
    REVIEWS["date"] = REVIEWS["date"].apply(lambda x: pd.to_datetime(x))

    # Filter the Lookup Dataset based on appNos
    appNos = []
    for drug in drugs:
        if drug in mapping:
            appNos.append(mapping[drug])
        else:
            appNos.append("")
    LABELS = labels[labels.applnNo.isin(appNos)].set_index("applnNo")
    EVENTS = events[events.applnNo.isin(appNos)].reset_index(drop=True)
    REACTIONS = reactions[reactions.applnNo.isin(
        appNos)].reset_index(drop=True)

    return appNos, REVIEWS, LABELS, EVENTS, REACTIONS


def temporalRating(reviewsDf, drugs):
    plot_df = reviewsDf[reviewsDf.drug.isin(drugs[:3])]
    plot_df = plot_df.groupby(["drug", "date"])["rating"].mean().reset_index()
    fig = px.line(plot_df, x="date", y="rating", color='drug')
    return fig


def sentimentAnalysis(reviewsDf, drugs):
    sentimentDf = reviewsDf.groupby(["drug", "sentiment"])[
        "usefulCount"].sum().reset_index()
    sentimentDf = sentimentDf.sort_values(
        by=["drug", "sentiment"], ascending=True)
    sentimentDf["sentiment"] = sentimentDf["sentiment"].apply(
        lambda x: "Positive" if x == 2 else ("Neutral" if x == 1 else "Negative"))
    fig = px.histogram(sentimentDf[sentimentDf.drug.isin(drugs)], x="sentiment", y="usefulCount",
                       color='drug', barmode='group')
    return fig


def getLabelCards(labels, drugs, appNos, nlp_token_class):
    cols = ["indications_and_usage", "contraindications",
            "pregnancy", "nursing_mothers", "pediatric_use"]
    names = ["DETAILED_DESCRIPTION", "DISEASE_DISORDER",
             "SIGN_SYMPTOM", "SIGN_SYMPTOM", "SIGN_SYMPTOM"]

    LABELCARDS = []
    for drug, appNo in zip(drugs, appNos):
        if appNo != "":
            temp = {"drug": drug}
            for col, name in zip(cols, names):
                try:
                    if str(labels.loc[appNo, col]) != "nan":
                        tags = pd.DataFrame(
                            nlp_token_class(labels.loc[appNo, col]))
                        tags = tags[((tags.entity_group == name) & (tags.score >= 0.8))].sort_values(
                            by="score", ascending=False)
                        if len(tags) > 0:
                            signs = " || ".join(
                                [item for item in tags.word.unique()[:3] if "type" not in item])
                            temp[col] = signs
                    else:
                        temp[col] = np.nan
                except KeyError:
                    continue
            LABELCARDS.append(temp)
    LABELCARDS = pd.DataFrame(LABELCARDS)
    return LABELCARDS


def getAdverseEvents(events, drugs, appNos):
    labels = ["Death", "Disabling",
              "Hospitalization", "Life Threatening Event"]
    events2 = events[events.event.isin(labels)].drop_duplicates()
    if len(events2) == 0:
        return "No adverse events found."
    appNosExisting = list(events2.applnNo.unique())
    drugs2 = []
    for i, appNo in enumerate(appNos):
        if appNo in appNosExisting:
            drugs2.append(drugs[i])
    appNos = appNosExisting
    fig = make_subplots(rows=1, cols=len(appNos), subplot_titles=tuple(drugs2), specs=[
                        [{'type': 'domain'} for _ in range(len(appNos))]])
    for i, appNo in enumerate(appNos):
        fig.add_trace(go.Pie(labels=labels, values=list(
            events2[events2.applnNo == appNo]["count"].values)), 1, i+1)
    fig.update_traces(hoverinfo='label+percent', textinfo='value+percent')
    fig.update_layout(
        title_text=f"Adverse Events reported over the years (If count < 50k, it is insignificant)")
    return fig


def getReactions(reactions, appNo, drug):

    ageMap = {
        1: "Neonate",
        2: "Infant",
        3: "Child",
        4: "Adolescent",
        5: "Adult",
        6: "Elderly"}

    sexMap = {1: "Male", 2: "Female"}

    weightMap = {"l": "Underweight", "n": "Normal", "h": "Overweight"}

    df = reactions[((reactions.applnNo == appNo) & (reactions["count"] > 7))]
    df["sex"] = df["sex"].apply(lambda x: sexMap[x])
    df["age"] = df["age"].apply(lambda x: ageMap[x])
    df["weight"] = df["weight"].apply(lambda x: weightMap[x])

    fig = px.treemap(df, path=[px.Constant(
        "all"), "sex", "age", "weight", "reaction"], values="count")

    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25),
                      title_text=f"Reactions for drug {drug}")
    return fig


def compareDrugs(REVIEWS, LABELS, EVENTS, drugs, appNos):
    REVIEWS_AGG = REVIEWS[REVIEWS.drug.isin(drugs)].groupby("drug").agg(
        AvgRating=("rating", "mean"),
        Usefulness=("usefulCount", "sum"),
        CommonSentiment=("sentiment", pd.Series.mode))

    PRECAUTIONS = LABELS[["contraindications"]]
    PRECAUTIONS["precautionsCount"] = PRECAUTIONS["contraindications"].apply(
        lambda x: len(str(x).split("||")) if str(x) != "nan" else np.nan)
    PRECAUTIONS = PRECAUTIONS[["precautionsCount"]]

    SERIOUSCASES = EVENTS[EVENTS.event ==
                          "Serious Cases"].drop_duplicates().set_index("applnNo")
    SERIOUSCASES = SERIOUSCASES[["count"]]

    reverseMapping = {}
    for drug, appNo in zip(drugs, appNos):
        if appNo != "":
            reverseMapping[appNo] = drug
    PRECAUTIONS = PRECAUTIONS.rename(index=reverseMapping)
    SERIOUSCASES = SERIOUSCASES.rename(index=reverseMapping)

    AGG_DATA = REVIEWS_AGG.merge(PRECAUTIONS, left_index=True, right_index=True, how="left").merge(
        SERIOUSCASES, left_index=True, right_index=True, how="left")
    AGG_DATA = AGG_DATA.reset_index()
    AGG_DATA.CommonSentiment = AGG_DATA.CommonSentiment.apply(
        lambda x: x[0] if type(x) == np.ndarray else x)

    AGG_DATA = AGG_DATA.fillna(0)
    
    AGG_DATA["AvgRating"] = AGG_DATA["AvgRating"]/10
    try:
        AGG_DATA["Usefulness"] = AGG_DATA["Usefulness"] / \
        (10**(int(round(np.log10(AGG_DATA["Usefulness"].mean())))))
    except OverflowError:
        AGG_DATA.loc[:,"Usefulness"] = 0
    AGG_DATA["CommonSentiment"] = AGG_DATA["CommonSentiment"].apply(
        lambda x: 3 if x == 0 else (6 if x == 1 else 10))
    try:
        AGG_DATA["count"] = AGG_DATA["count"] / \
        (10**(int(round(np.log10(AGG_DATA["count"].mean())))))
    except OverflowError:
        AGG_DATA.loc[:,"count"] = 0

    AGG_DATA = AGG_DATA.rename(columns={"index": "drug",
                                        "count": "Serious Cases",
                                        "CommonSentiment": "Sentiment",
                                        "precautionsCount": "Precautions",
                                        "AvgRating": "Rating"})

    categories = AGG_DATA.columns[1:]
    fig = go.Figure()
    for drug in reverseMapping.values():
        fig.add_trace(go.Scatterpolar(
            r=AGG_DATA[AGG_DATA.drug == drug].values[0][1:],
            theta=categories,
            fill='toself',
            name=drug))

    fig.update_polars(gridshape='linear', bgcolor="#0E1117")
    return fig


def plot(response, setupData):
    drugs = getDrugs(response)
    appNos, REVIEWS, LABELS, EVENTS, REACTIONS = getData(
        drugs, setupData["reviews"], setupData["labels"], setupData["reactions"], setupData["events"], setupData["mapping"])

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Trend Analysis", "Sentiment Analysis", "Precautionary Labels", "Adverse Events", "Reactions", "Drug Compare"])

    with tab1:
        st.plotly_chart(temporalRating(REVIEWS, drugs),
                        use_container_width=True)

    with tab2:
        st.plotly_chart(sentimentAnalysis(REVIEWS, drugs),
                        use_container_width=True)

    with tab3:
        labelCards = getLabelCards(LABELS, drugs, appNos, setupData["tokens"])
        if len(labelCards.columns) == 1:
            st.write("No labels found.")
        else:
            st.dataframe(labelCards)

    with tab4:
        adverse_events_chart = getAdverseEvents(EVENTS, drugs, appNos)
        if isinstance(adverse_events_chart, str):
            # Display message if no adverse events found
            st.write(adverse_events_chart)
        else:
            st.plotly_chart(adverse_events_chart, use_container_width=True)

    with tab5:
        appNo = REACTIONS.applnNo.value_counts().index
        if len(appNo) > 0:
            appNo = appNo[0]
            drug = drugs[appNos.index(appNo)]
            st.plotly_chart(getReactions(REACTIONS, appNo, drug), use_container_width=True)
        else:
            appNo = ""
            drug = ""
            st.write("No adverse reactions found.")

    with tab6:
        # st.dataframe(compareDrugs(REVIEWS, LABELS, EVENTS, appNos, drugs))
        st.plotly_chart(compareDrugs(REVIEWS, LABELS, EVENTS,
                        drugs, appNos), use_container_width=True)
