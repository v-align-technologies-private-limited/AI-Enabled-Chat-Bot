import configparser
import openai
import pandas as pd
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pinecone
import os
from pinecone import Pinecone, ServerlessSpec
import spacy
nlp = spacy.load("en_core_web_sm")
class Config:
    config = configparser.ConfigParser()
    config.read('config2.ini')

    # OpenAI settings
    OPENAI_API_KEY = config['openai']['api_key']
    openAI_model = config['openai']['model']

    # Database settings
    DATABASE_HOST = config['database']['host']
    DATABASE_USERNAME = config['database']['username']
    DATABASE_PASSWORD = config['database']['password']
    PORT = config.getint('database', 'port')

    # Pinecone settings
    PINECONE_API_KEY = config['pinecone']['api_key']
    INDEX_NAME = config['pinecone']['index_name']

    # Model settings
    MODEL_NAME = config['model']['name']
class Initialize_config:
    def __init__(self):
        print("Initial configuration is going on")
        self.embedding_model=AutoModel.from_pretrained(Config.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.pinecone_index=None
        self.openai_model=None
        self.prompt_template = None
    def __del__(self):
        pass
    def assign_pinecone_index(self):
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        if Config.INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=Config.INDEX_NAME,
                dimension=768,
                metric='dotproduct',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.pinecone_index=pc.Index(Config.INDEX_NAME)
    def process_openAI_model(self):
        openai.api_key=Config.OPENAI_API_KEY
        self.openai_model = ChatOpenAI(
        openai_api_key=openai.api_key,
        model_name=Config.openAI_model,
        temperature=0.7,
        max_tokens=150)
    def set_prompt_template(self):
        template = """## Knowledge Base:
            {knowledge_base}

            ## Database Schema:
            {database_schema}

            ## Question:
            {question}

            ## Answer:
            """
        self.prompt_template=ChatPromptTemplate.from_template(template)
