import pandas as pd
import os
from multiprocessing import Pool
import numpy as np
import requests
import warnings
import pickle
warnings.filterwarnings("ignore")

def getReactions(appNo):
    reactions = []

    baseUrl = 'https://api.fda.gov/drug/event.json?count=patient.patientweight'
    baseFilter = f'_exists_:patient.patientweight+AND+_exists_:patient.patientsex+AND+_exists_:patient.patientagegroup+AND+patient.drug.openfda.application_number:{appNo}'
    
    for s in ["1", "2"]:
        for a in ["1", "2", "3", "4", "5", "6"]:
            sexFilter = f'patient.patientsex:{s}'
            ageFilter = f'patient.patientagegroup:{a}'
            url = f'{baseUrl}&search={baseFilter}+AND+{sexFilter}+AND+{ageFilter}&limit=1000'
            response = requests.get(url)
            
            if response.status_code == 200:
                weights = [item["term"] for item in response.json()["results"]]
                weightFilter1 = ""
                weightFilter2 = ""
                weightFilter3 = ""

                for weight in weights:
                    if weight < 45:
                        weightFilter1 += f'"{weight}"+OR+'
                    elif weight > 95:
                        weightFilter3 += f'"{weight}"+OR+'
                    else:
                        weightFilter2 += f'"{weight}"+OR+'

                weightFilter1 = f"patient.patientweight:({weightFilter1[:-4]})" if weightFilter1 != "" else ""
                weightFilter2 = f"patient.patientweight:({weightFilter2[:-4]})" if weightFilter2 != "" else ""
                weightFilter3 = f"patient.patientweight:({weightFilter3[:-4]})" if weightFilter3 != "" else ""
                
                baseUrl2 = 'https://api.fda.gov/drug/event.json?count=patient.reaction.reactionmeddrapt.exact'
                for wf,name in zip([weightFilter1, weightFilter2, weightFilter3], ["l", "n", "h"]):
                    if wf != "":
                        url2 = f'{baseUrl2}&search={baseFilter}+AND+{sexFilter}+AND+{ageFilter}+AND+{wf}&limit=3'
                        response2 = requests.get(url2).json()["results"]
                        for item in response2:
                            reactions.append({
                                "age": a,
                                "sex": s,
                                "weight": name,
                                "reaction": item["term"],
                                "count": item["count"]
                            })
            elif response.status_code == 429:
                print("Exceeded Rate limit")
                return pd.DataFrame()      
            else:
                continue

    reactions = pd.DataFrame(reactions)
    if len(reactions) > 0:
        reactions.loc[:,"applnNo"] = appNo
    return reactions

def processBatch(appNos, iteration, batchNum, batchSize):
    reactions = pd.DataFrame()
    start = batchNum*batchSize
    end = (batchNum+1)*batchSize
    errors = []
    for i,appNo in enumerate(appNos[start:end]):
        try:
            reactions = pd.concat([reactions, getReactions(appNo)], ignore_index=True)
        except:
            errors.append(appNo)
        if i%10 == 0:
            print(f"Batch:{batchNum} || Completed {i} items")
    if len(reactions) != 0:
        print(f"Batch:{batchNum} || {reactions['applnNo'].nunique()} Drugs")
    else:
        print(f"Batch:{batchNum} || 0 Drugs")
    reactions.to_csv(f"./LookupData/ReactionsData/Reactions_{iteration}_{batchNum}.csv")

    with open(f"./Errors_{iteration}_{batchNum}.pickle","wb") as file:
        pickle.dump(errors, file)

def main():
    with open("./AppNos.pickle", "rb") as file:
        appNos = pickle.load(file)
    
    # total_calls = 400
    iteration = 5
    appNos = appNos[2000:2250]

    batchSize = 50
    batches = len(appNos) // batchSize

    with Pool(os.cpu_count()) as p:
        p.starmap(processBatch, [(appNos, iteration, i, batchSize) for i in range(batches)])

if __name__ == "__main__":
    main()