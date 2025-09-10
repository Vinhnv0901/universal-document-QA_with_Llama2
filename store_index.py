from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Chroma






extracted_data=load_pdf_file(data='data/')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
persist_directory = 'db'


vectordb = Chroma.from_documents(documents=text_chunks,
                                 embedding=embeddings,
                                 persist_directory=persist_directory)

vectordb.persist()


