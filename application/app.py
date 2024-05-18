from flask import Flask, request, jsonify
from flasgger import Swagger
from flasgger.utils import swag_from
from langchain import PromptTemplate
# from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
import re

import os

os.environ['CURL_CA_BUNDLE'] = ''

app = Flask(__name__)
swagger = Swagger(app)

"""
local_llm = "models/BioMistral-7B.Q4_K_M.gguf"
 
llm = LlamaCpp(
    model_path=local_llm,
    temperature=0.1,
    max_tokens=2048,
    n_ctx=2048,
    top_p=1
)
"""

groq_api_key = "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"

groq_llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)

print("Initialised LLM")

prompt_template = """
Use the provided drug information CSV file to recommend medicines for the user's condition.
If you don't know the answer or can't find relevant medicines, say so politely.
 
Context: {context}
Question: {question}
 
To recommend medicines, follow these steps:
1. Search the CSV file for entries where the "condition" column matches the user's condition.
2. If multiple entries are found, include all relevant medicines and their details in your recommendation.
3. If no relevant entries are found, politely mention that no recommended medicines were found for the given condition.
 
Present your recommendation in a concise and well-organized manner, using bullet points or numbered lists if applicable.
Remember to give top 3 recommendations for the given user's "condition" and each recommendation should have dosage, side effects, rating. Refer to the example recommendation format below.
Ensure that the drug name is available in CSV
 
Example recommendation format:
For the condition "headache", the recommended medicines are:
- Medicine "drugName"
    - Dosage: [Dosage]
    - Side Effects: [Side Effects]
    - Rating: [rating]
"""

embeddings = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings")

url = "http://qdrant:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings,
            collection_name="drugs_db")

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k": 4})


def extract_medicine_names(response):
    pattern = r'Medicine "([^"]+)"'
    matches = re.findall(pattern, response)
    return matches


@app.route('/')
def index():
    return 'Backend is running'


@app.route("/get_response", methods=["POST"])
@swag_from("swagger_config.yml")
def get_response():
    """
    Retrieve a response based on the provided query.
    ---
    parameters:
      - name: query
        in: formData
        type: string
        required: true
        description: The query to retrieve a response for.
    responses:
      200:
        description: Successful response retrieval.
    """
    query = request.form['query']
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=groq_llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = {
        "answer": answer,
    }
    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
