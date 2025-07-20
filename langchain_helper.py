import streamlit as st

from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

vectordb_file_path = "faiss_index"

embeddings = HuggingFaceEmbeddings()

def create_vector_db(csv_file_path='codebasics_faqs.csv'):
    """
    Create vector database from CSV file
    Args:
        csv_file_path: Path to CSV file (defaults to codebasics_faqs.csv)
    """
    loader = CSVLoader(file_path=csv_file_path, source_column='prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain(user_query):
    # load vector db from local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)

    # create retriever for querying vector database
    retriever = vectordb.as_retriever()

    # set the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0,
        max_tokens=None,
        timeout=None,
    )

    # set up prompt template
    prompt_template = """Below you will be given context and a question. Generate and answer to the question based on the context only. And while answering imagine you are a customer service person. So be polite and well spoken.
    If the context does not contain enough informaiton regarding the question, and you cannot provide an answer with confidence, just say that you don't know what the answer is but DO NOT make stuff up.
    In the event that this happens and you have to answer with a "I don't know", don't just say "I don't know" because it sounds bland. Reply more like a polite human in this case

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['context', 'question']
    )

    # 6 Set up chain invocation
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":PROMPT})

    raw_llm_response = chain(user_query)

    return raw_llm_response

if __name__ == "__main__":
    user_query = str(input("Ask a question:\n"))
    raw_llm_response = get_qa_chain(user_query)
    print(raw_llm_response['result'])