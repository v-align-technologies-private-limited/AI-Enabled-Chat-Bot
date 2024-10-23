import pandas as pd
class Schema_manager:
    def __init__(self,conn,query,schema):
        self.conn=conn
        self.query=query
        self.schema_df=schema
        self.schema_str = ""
    def __del__(self):
        pass
    def fetch_schema_with_data_types(self):
        try:
            self.schema_df=pd.read_sql(self.query, self.conn)
        except Exception as e:
            print(f"Error fetching schema with data types: {e}")
            raise
    def format_schema(self):
        self.schema_str = ""
        grouped = self.schema_df.groupby('table_name')
        for table_name, group in grouped:
            columns = ', '.join([f"{row['column_name']} ({row['data_type']})" for _, row in group.iterrows()])
            self.schema_str += f"{table_name}: {columns}\n"
