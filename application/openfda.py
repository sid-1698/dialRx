import pandas as pd
import os
import numpy as np
import requests

def getResponse(url, count=False):
    """
    Makes a GET request to the specified URL and returns the response data as a pandas DataFrame.

    Parameters:
    - url (str): The URL to which the GET request is made.
    - count (bool, optional): If True, only the total count of items is returned. 
      If False (default), the function returns the total count of results and the results in a DataFrame.

    Returns:
    - tuple: A tuple containing the DataFrame with the response data and the total count of items.
             If there's an error in getting the response, it returns an empty DataFrame and None for the total count.

    Note:
    - This function is intended to be used with APIs that return JSON data compatible with the expected response structure.
    - In case of a non-200 response status code, it prints an error message.
    """
    response = requests.get(url)
    if response.status_code == 200:
        response = response.json()
        if count == False:
            totalCount = response["meta"]["results"]["total"]
        else:
            totalCount = 1
        df = pd.DataFrame(response["results"])   
        return df, totalCount
    else:
        print(f"Error in getting response")
        return pd.DataFrame(), -1
    
def getRecentBrandNames(drug):
    """
    Fetches recent brand names for a specified drug by making a GET request to the FDA's API,
    sorting by listing expiration date in descending order.

    Parameters:
    - drug (str): The drug name to search for in the brand name base.

    Returns:
    - tuple: A tuple containing a DataFrame with the fetched data and a list of application IDs (applnIds).
             If the request is successful, it returns the data and application IDs, otherwise returns an empty DataFrame and an empty list.

    Note:
    - The function aims to fetch the first 1000 responses sorted by listing expiration date,
      dropping duplicates based on the application number, and returns the most recent entries.
    - It prints the number of fetched responses out of the total available responses.
    """
    baseUrl = "https://api.fda.gov/drug/ndc.json?limit=1000&sort=listing_expiration_date:desc"
    filter = f"brand_name_base:*{drug}*+AND+finished:true"
    url = f"{baseUrl}&count=application_number&search={filter}"
    df, status = getResponse(url, True)
    if status == 1:
        applnIds = df.term.values[:5]
        filter = f"application_number:"
        for id in applnIds:
            filter += f"{id}+OR+"
        filter = filter[:-4]
        url = f"{baseUrl}&search={filter}"
        df, cnt = getResponse(url)
        print(f"Fetched {min(1000,cnt)} responses out of {cnt} responses...")
        df = df.sort_values(by="listing_expiration_date", ascending=False).drop_duplicates(subset=["application_number"], keep="first").reset_index(drop=True)
        return df, applnIds
    else:
        return pd.DataFrame(), []
    
def getBrandLabels(applnIds):
    """
    Fetches brand label information for a list of application IDs by making GET requests to the FDA's API.
    The response is sorted by effective time in descending order.

    Parameters:
    - applnIds (list): A list of application IDs for which to fetch brand label information.

    Returns:
    - DataFrame: A pandas DataFrame containing the brand label information associated with each application ID.

   Note:
    - The function queries the FDA's API for label information based on provided application IDs.
    - It ensures that each returned label is unique per application ID by dropping duplicates.
    - The number of fetched responses is printed to indicate the scope of the returned data.
    """

    baseUrl = "https://api.fda.gov/drug/label.json?limit=1000&sort=effective_time"
    filter = f"openfda.application_number:"
    for id in applnIds:
        filter += f"{id}+OR+"
    filter = filter[:-4]
    url = f"{baseUrl}&search={filter}"
    df,cnt = getResponse(url)
    print(f"Fetched {min(1000,cnt)} responses out of {cnt} responses...")
    df["applnNo"] = df["openfda"].apply(lambda x: x["application_number"])
    df = df.drop_duplicates(subset=["applnNo"], keep="first")
    return df

def getEventsCount(applnIds):
    """
    Fetches the count of various event types (e.g., serious, death, disabling) associated with a list of application IDs from the FDA's adverse event reporting system.

    Parameters:
    - applnIds (list of str): A list of application IDs to query for adverse event data.

    Returns:
    - DataFrame: A pandas DataFrame containing the pivot table of event counts, indexed by event type and with columns for each term related to the events.

    Note:
    - This function performs multiple API calls to gather counts for different event types associated with the specified application IDs.
    - It aggregates the data into a pivot table format, normalizing the counts by the total number of events to facilitate comparison.
    """
    baseUrl = "https://api.fda.gov/drug/event.json?limit=1000"
    filter = f"patient.drug.openfda.application_number:("
    for id in applnIds:
        filter += f"{id}+OR+"
    filter = filter[:-4] + ")"

    count = {}
    for col in ["serious", "seriousnessdeath", "seriousnessdisabling", "seriousnesshospitalization", "seriousnesslifethreatening"]:
        url = f"{baseUrl}&count={col}&search={filter}"
        response = requests.get(url)
        count[col] = response.json()["results"]

    res = []
    for k,val in count.items():
        for item in val:
            term = item["term"]
            count = item["count"]
            res.append({
                "term": term,
                "count": count,
                "event": k
            })
    res = pd.DataFrame(res)
    res = pd.pivot_table(res, index="event", values="count", columns="term")
    for col in res.columns:
        res[col] = res[col]/res.sum(axis=1)

    return res

def getAdverseReactions(applnIds):
    """
    Fetches and processes adverse reaction data for a list of application IDs from the FDA's API.
    The function queries the API for adverse reactions associated with the provided application IDs,
    and augments the reaction data with patient age group, weight category, and sex.
    
    The weight category is derived based on patient weight: 
    - 1 for weights over 90 kg,
    - -1 for weights under 45 kg, 
    - 0 for weights in between.

    Parameters:
    - applnIds (list of str): A list of application IDs for which to fetch adverse reaction data.

    Returns:
    - DataFrame: A pandas DataFrame containing detailed information about adverse reactions, grouped by 
      reaction outcome, age, weight category, and sex. Only reactions with more than one occurrence are returned.

    Note:
    - This function provides insights into adverse reactions, potentially revealing patterns based on patient demographics.
    - It processes the response to extract detailed adverse reaction information, including the medical term for the reaction, and groups the data for analysis.
    - The function prints the number of fetched responses to indicate the scope of the data returned.
    """
    baseUrl = "https://api.fda.gov/drug/event.json?limit=1000&sort=receiptdate:desc"
    filter = f"patient.drug.openfda.application_number:("
    for id in applnIds:
        filter += f"{id}+OR+"
    filter = filter[:-4] + ")"
    url = f"{baseUrl}&search={filter}"

    df, cnt = getResponse(url)
    print(f"Fetched {min(1000,cnt)} responses out of {cnt} responses...")
    patients = list(df["patient"].values)
    df = pd.DataFrame(patients)
    df = df.dropna().reset_index(drop=True)
    reactions = []
    for i in range(len(df)):
        item = df.reaction[i]
        for react in item:
            react["age"] = df["patientagegroup"][i]
            react["weight"] = 1 if float(df["patientweight"][i]) > 90 else (-1 if float(df["patientweight"][i]) < 45 else 0)
            # react["weight"] = float(df["patientweight"][i])
            react["sex"] = df["patientsex"][i]
            reactions.append(react)
    reactions = pd.DataFrame(reactions)
    reactions = reactions.groupby(["reactionoutcome", "age", "weight", "sex"])["reactionmeddrapt"].value_counts().reset_index().drop_duplicates(["reactionoutcome", "age", "weight", "sex"]).reset_index(drop=True)
    reactions = reactions[reactions["count"] > 1]
    return reactions