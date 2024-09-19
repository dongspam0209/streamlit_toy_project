import os
import json
import traceback
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import pymysql

pymysql.install_as_MySQLdb()

load_dotenv()
from sqlalchemy import create_engine

engine = create_engine(
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}?charset={os.getenv('DB_CHARSET')}",
    echo=True
)

def load_dataframe(df, table_name, conn):
    """
    dataframe → table

    Args:
        df (dataframe): table로 저장할 dataframe
        table_name (str): 저장할 table의 table명
        conn (pymysql.connect): db connection 의미
    """	
    try:
        df.to_sql(table_name, con=conn, if_exists="append", index=False)
    except:
        # traceback 메시지를 문자열로 반환
        print(traceback.format_exc())



final_topic_df=pd.read_csv('topic_table.csv',encoding='utf-8',header=0)
final_keyword_df=pd.read_csv('keyword_table.csv',encoding='utf-8',header=0)
final_speaker_df=pd.read_csv('speaker_table.csv',encoding='utf-8',header=0)
final_utterances_info_df=pd.read_csv('utterances_info_table.csv',encoding='utf-8',header=0)
final_utterances_df=pd.read_csv('utterances_table.csv',encoding='utf-8',header=0)

print(final_topic_df.head())
print(final_keyword_df.head())
print(final_speaker_df.head())
print(final_utterances_info_df.head())
print(final_utterances_df.head())

load_dataframe(final_topic_df,"topic",engine) #topic table에 저장
load_dataframe(final_keyword_df,"keyword",engine) #keyword table에 저장

final_speaker_df.drop_duplicates(['id'],inplace=True)
final_speaker_df.dropna(inplace=True)
load_dataframe(final_speaker_df,"speaker",engine) #speaker table에 저장
load_dataframe(final_utterances_info_df,"utterances_info",engine) #utterances info table에 저장
load_dataframe(final_utterances_df,"utterances",engine) #utterances table에 저장