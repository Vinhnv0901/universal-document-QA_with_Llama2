from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from transformers import pipeline
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain import HuggingFacePipeline
app = Flask(__name__)

load_dotenv()
token = os.environ["HUGGINGFACEHUB_API_TOKEN"]
login(token=token)



embeddings = download_hugging_face_embeddings()



# Load lại vectordb đã lưu
persist_directory = "db"
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":3})




model = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="auto",
    quantization_config=bnb_config,
    token=token
)
pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0.4})


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
