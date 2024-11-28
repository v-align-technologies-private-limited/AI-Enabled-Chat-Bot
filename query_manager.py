import pandas as pd
import re
import json
import numpy as np
import os
from fuzzywuzzy import fuzz
from datetime import datetime
import spacy
nlp = spacy.load("en_core_web_sm")
class Determine_querry_type:
    def __init__(self,schema_df, threshold=60):
        self.query_type="knowledge"
        self.user_query=''
        self.schema_df=schema_df
        self.threshold=threshold
    def __del__(self):
        pass
    def determine_query_type(self,user_query):
        self.user_query=user_query
        if isinstance(self.user_query, dict):
            self.query_type='database'
            return
        user_query_lower = self.user_query.lower()
        table_names = self.schema_df['table_name'].str.lower().unique()
        column_names = self.schema_df['column_name'].str.lower().unique()
        
        # Function to check fuzzy match
        def is_fuzzy_match(query, options):
            for option in options:
                if fuzz.partial_ratio(query, option) >= self.threshold:
                    return True
            return False
        # Function to check substring match
        def is_partial_match(query, options):
            for option in options:
                if query in option or option in query:
                    return True
            return False
        
        # Check if user query matches any table or column name
        if is_fuzzy_match(user_query_lower, table_names) or \
           is_fuzzy_match(user_query_lower, column_names) or \
           is_partial_match(user_query_lower, table_names) or \
           is_partial_match(user_query_lower, column_names):
            self.query_type="database"
        else:
            self.query_type="knowledge"
    def is_entity_present(self,schema_entities,user_text):
        user_text_lower = user_text.lower()
        for entity in schema_entities:
            if entity.lower() in user_text_lower:
                return True  
        return False
    def contains_date_related_text(self,user_input):
        # Current year and month for comparison
        current_year = datetime.now().year
        current_month = datetime.now().strftime("%B")
        
        # Patterns to match date, month, year, and relative terms
        date_pattern = r'\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})\b'
        month_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December|' \
                        r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'
        year_pattern = r'\b(19|20)\d{2}\b'
        time_pattern = r'\b\d+\s*(or\s*(more|fewer|less))?\s*(days?|weeks?|months?|years?)\b'
        relative_terms_pattern = r'\b(this month|this year|last month|last year|next month|next year)\b'
        # Find matches
        if (re.search(date_pattern, user_input) or 
            re.search(month_pattern, user_input, re.IGNORECASE) or 
            re.search(year_pattern, user_input) or 
            re.search(time_pattern, user_input) or
            re.search(relative_terms_pattern, user_input, re.IGNORECASE)):
            return True
        return False
    def named_entity_recognition(self,text):
        doc = nlp(text)
        # Check for entities that are either PERSON, ORG, or GPE
        for entity in doc.ents:
            if entity.label_ in ("PERSON", "ORG", "GPE"):
                return True
        return False
