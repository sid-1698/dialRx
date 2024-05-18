import streamlit as st
import requests
import json
import re
import pandas as pd
import altair as alt
from st_circular_progress import CircularProgress
import os
from PlotHelpers import setup, plot

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(current_dir, "..")
os.chdir(project_dir)

# Page config
st.set_page_config(page_title="DialRx", page_icon=f"{project_dir}/data/logo.png")

def reset_session():
    # Display HTML to reload the page
    st.markdown("<meta http-equiv='refresh' content='0'>",
                unsafe_allow_html=True)


def displayRecommendations(response, REVIEWS):
    recommendations = response.split("\n\n")
    if len(recommendations) <= 2:
        st.write(
            "No recommendations found, please try again or try searching for a different condition")
        return False
    for recom in recommendations[1:-1]:
        try:
            drug = re.search(r'Medicine\s*\"([^\"]+)\"', recom).group(1)
            drug = drug.split(" ")[0].split("/")[0].lower()
        except AttributeError:
            # Check
            pass
            return
        with st.container():
            st.markdown(recom)
            with st.expander(f"Reviews of {drug}"):
                positiveReviews = REVIEWS[((REVIEWS.drug == drug) & (REVIEWS.sentiment == 2))].sort_values(
                    by="usefulCount", ascending=False)["review"].values[:5]
                negativeReviews = REVIEWS[((REVIEWS.drug == drug) & (REVIEWS.sentiment == 0))].sort_values(
                    by="usefulCount", ascending=False)["review"].values[:5]
                positiveReviews = [review.split(". ")[0]
                                   for review in positiveReviews]
                negativeReviews = [review.split(". ")[0]
                                   for review in negativeReviews]
                if ((len(positiveReviews) == 0) and (len(negativeReviews) == 0)):
                    st.write("No Reviews Found")
                    continue
                else:
                    st.write("Positive Reviews")
                    for review in positiveReviews:
                        st.markdown('''\n :green[{}]'''.format(review))
                    st.write("Negative Reviews")
                    for review in negativeReviews:
                        st.markdown('''\n :red[{}]'''.format(review))
    return True


st.title("DialRx Bot")

st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        <p>&copy; 2024 DialRx. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if 'previous_inputs' not in st.session_state:
    st.session_state.previous_inputs = []

st.sidebar.image(f"{project_dir}/data/logo.png", width=200)
st.sidebar.title("Chatbox")

setupData = setup()

st.session_state.messages = []  # Clear previous search results

user_input = st.sidebar.text_input("Enter your diagnosis..", key="user_input")
if st.sidebar.button("Send"):
    with st.spinner('Fetching response...'):
        formData = {'query': user_input}
        response = requests.post(
            'http://localhost:5001/get_response', data=formData)
        if response.ok:
            data = response.json()
            st.session_state.messages = []
            isVisualised = displayRecommendations(
                data["answer"], setupData["reviews"])
            if isVisualised:
                plot(data, setupData)
            else:
                pass
            st.session_state.previous_inputs.append(user_input)
        else:
            st.sidebar.error("Failed to get response")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "assistant":
            st.success(message["content"])
        else:
            st.info(message["content"])

# Dropdown list for previous inputs
if st.session_state.previous_inputs:
    selected_input_index = st.sidebar.selectbox(
        "Previous Inputs", options=st.session_state.previous_inputs, index=len(st.session_state.previous_inputs) - 1)

    if user_input != selected_input_index:
        user_input = selected_input_index
        formData = {'query': user_input}
        with st.spinner('Fetching response...'):
            response = requests.post(
                'http://localhost:5001/get_response', data=formData)
            if response.ok:
                data = response.json()
                st.session_state.messages = []
                isVisualised = displayRecommendations(
                    data["answer"], setupData["reviews"])
                if isVisualised:
                    plot(data, setupData)
                else:
                    pass
            else:
                st.sidebar.error("Failed to get response")

# Reset session button
if st.button("Reset"):
    reset_session()
    st.write("State has been reset.")