import psycopg2
from initial_config import *
import pandas as pd
import re
import json
import numpy as np
from datetime import datetime
class DB_Manager:
    def __init__(self):
        self.connection = None
        self.results=''
    def __del__(self):
        pass

    def connect(self,DATABASE_DB):
        try:
            self.connection = psycopg2.connect(
                dbname=DATABASE_DB,
                user=Config.DATABASE_USERNAME,
                password=Config.DATABASE_PASSWORD,
                host=Config.DATABASE_HOST,
                port=Config.PORT
            )
            return self.connection
        except psycopg2.Error as e:
            print(f"Error connecting to the database: {e}")
            raise
    def close_connection(self):
        self.connection.close()
    def execute_sql_query(self,sql_query):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql_query)
                results = cursor.fetchall()
                self.results=results
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            self.results=[]
