from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone  # ✅ Corrected Pinecone import
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from src.prompt import *
import os
from pinecone import Pinecone  # ✅ Ensure you are using `pinecone-client`
from langchain.vectorstores import Pinecone


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"


# ✅ Correct import

docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = GoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", temperature=0.4, max_output_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running. Send a POST request to /chat with 'msg' parameter."})


"""@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])"""

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()  # Get JSON input
        if not data or "msg" not in data:
            return jsonify({"error": "Missing 'msg' parameter"}), 400

        user_message = data["msg"]
        print("User Input:", user_message)

        response = rag_chain.invoke({"input": user_message})  # Call AI Model
        answer = response.get("answer", "No response available.")

        print("Response:", answer)
        return jsonify({"response": answer})  # Return AI response as JSON

    except Exception as e:
        return jsonify({"error": str(e)}), 500
 # Directly returning the answer as plain text




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)