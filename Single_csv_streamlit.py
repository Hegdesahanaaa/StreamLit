import os
import markdown
import html2text
import fnmatch
import pickle

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PDFMinerLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI   
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
import streamlit as st
import pandas as pd
import openai

from classes import get_primer,format_question,run_request


st.set_page_config(page_icon="chat2vis.png",layout="wide",page_title="ChatPlot")
st.markdown("<h1 style='text-align: center; font-weight:bold; color:red;font-family:comic sans ms; padding-top: 0rem;'> \
            ChatPlot</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>Creating Visualisations using Natural Language \
            with ChatGPT</h2>", unsafe_allow_html=True)


# List to hold datasets
if "datasets" not in st.session_state:
    datasets = {}
    st.session_state["datasets"] = datasets
else:
    # use the list already loaded
    datasets = st.session_state["datasets"]

with st.sidebar:
    # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    dataset_container = st.empty()

    # Add facility to upload a dataset


    uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
    index_no=0
    if uploaded_file is not None:        
        # Read in the data, add it to the list of available datasets
        file_name = uploaded_file.name[:-4].capitalize()
        datasets[file_name] = pd.read_csv(uploaded_file)
        # Default for the radio buttons
        index_no = len(datasets)-

#langchain code
key=os.environ["OPENAI_API_KEY"] = ""

def prepare_Documents(input_Documents):
    # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the
    # token size limits.
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1200,
        chunk_overlap=128,
        length_function=len,
    )
    return text_splitter.split_documents(input_Documents)

if uploaded_file is not None:
    csv_file=uploaded_file.name
    loader = CSVLoader(csv_file).load()
    prepared_data = prepare_Documents(loader)
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    faiss_support_store = FAISS.from_documents(prepared_data, embeddings)
    with open("faiss_support_store.pickle", "wb") as handler:
        pickle.dump(faiss_support_store, handler)

    support_template = """
        Note:
        1.Be polite when addressing user queries.
        2.Provide context-based responses.
        3.If a question cannot be answered with the available information, respond with "I apologize, but that falls outside of my current scope of knowledge."
        4.Locate the answer within the given information.
        5.Keep the response within 3000 tokens.
        
        {context}
        
        Question: {question}"""
    
    SUPPORT_PROMPT = PromptTemplate(
    template=support_template, input_variables=["context", "question"] )
    
    chain_type_kwargs = {"prompt": SUPPORT_PROMPT}
    qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    chain_type="stuff", 
    retriever=faiss_support_store.as_retriever(), 
    chain_type_kwargs=chain_type_kwargs)

    # Text area for query
    question = st.text_area(":eyes: What would you like to visualise?",height=10)
    go_btn = st.button("Go...")
    
    result = qa({"query": question})
    if question:
        st.text(result)


# Display the datasets in a list of tabs
# Create the tabs

if datasets :
    tab_list = st.tabs(datasets.keys())
    # Load up each tab with a dataset
    for dataset_num, tab in enumerate(tab_list):
        with tab:
        # Can't get the name of the tab! Can't index key list. So convert to list and index
            dataset_name = list(datasets.keys())[dataset_num]
            st.subheader(dataset_name)
            st.dataframe(datasets[dataset_name],hide_index=True)


# Insert footer to reference dataset origin  
footer="""<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;text-align: center;}</style><div class="footer">
<p> <a style='display: block; text-align: center;'> Datasets courtesy of NL4DV, nvBench and ADVISor </a></p></div>"""
st.caption("Datasets courtesy of NL4DV, nvBench and ADVISor")

# Hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
