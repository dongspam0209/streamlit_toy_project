import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import os
import json
import traceback
import pandas as pd
import matplotlib as mpl
from tqdm import tqdm
from dotenv import load_dotenv
import pymysql
from sqlalchemy import create_engine

pymysql.install_as_MySQLdb()

load_dotenv()

st.secrets
engine = create_engine(
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}?charset={os.getenv('DB_CHARSET')}",
    echo=True
)

db_user = st.secrets["database"]["user"]
db_password = st.secrets["database"]["password"]
db_host = st.secrets["database"]["host"]
db_port = st.secrets["database"]["port"]
db_name = st.secrets["database"]["database"]

engine = create_engine(
    f"mysql+pymysql://{db_user}:{db_password}@"
    f"{db_host}:{db_port}/{db_name}",
    echo=True
)

@st.cache_data
def load_data():
    df_topic = pd.read_sql("select * from topic", engine)
    df_keyword = pd.read_sql("select * from keyword", engine)
    df_speaker = pd.read_sql("select * from speaker", engine)
    df_utterances_info = pd.read_sql("select * from utterances_info", engine)
    df_utterances = pd.read_sql("select * from utterances", engine)

    return df_topic,df_keyword,df_speaker,df_utterances_info,df_utterances





#### df utterances_info preprocessing
df_topic,df_keyword,df_speaker,df_utterances_info,df_utterances=load_data()

keyword_dic={k:v for k,v in zip(df_keyword['id'],df_keyword['keyword'])}
topic_dic={k:v for k,v in zip(df_topic['id'],df_topic['topic'])}

df_utterances_info['topic_id']=df_utterances_info['topic_id'].map(topic_dic)
df_utterances_info['keyword_id']=df_utterances_info['keyword_id'].map(keyword_dic)

####

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] =False

st.title("Multi-turn SNS data")


info_topic_group = df_utterances_info['topic_id'].value_counts() # number of topics
info_topic_per_group = df_utterances_info['topic_id'].value_counts() / len(df_utterances_info) # percentage of topic_id

# 1. Topic distribution
st.header("Topic Distribution")

fig, ax = plt.subplots(figsize=(15, 8))
bars = ax.bar(info_topic_group.index, info_topic_group,
              edgecolor='black',linewidth=1,color='royalblue')

for i,bar in enumerate(bars):
    width=bar.get_width()
    height=bar.get_height()
    x,y=bar.get_xy()
    percent=info_topic_per_group.iloc[i]
    plt.text(x+width/2,
             height+700,
             f'{percent:.2%}',
             ha='center',
             weight='bold')

ax.set_xticks(range(len(info_topic_group.index)))    
ax.set_xticklabels(info_topic_group.index, rotation=60, fontsize=14)  # fontsize를 조정하여 글자 크기 설정
ax.set_ylabel("Count",fontsize=14)

st.pyplot(fig)

# 2. Keyword distribution
st.header("Keywords per Topic")
selected_topic = st.selectbox("Select Topic:", df_utterances_info['topic_id'].unique())

# plot keyword distribution about selected topic
if selected_topic:
    selected_data = df_utterances_info[df_utterances_info['topic_id'] == selected_topic]
    keyword_counts = selected_data['keyword_id'].value_counts().head(5)  # 상위 5개 키워드만 표시
    keyword_counts.sort_values(inplace=True)
    # 선택된 토픽에 대한 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(keyword_counts.index, keyword_counts.values,
            color='royalblue', edgecolor='black', linewidth=1)
    ax.set_xlabel("Count",fontsize=14)
    ax.set_title(f"Top 5 Keywords for Topic : {selected_topic}")
    ax.spines[['top', 'right']].set_visible(False)
    plt.grid(True, axis='x',linestyle='--')

    st.pyplot(fig)

# 3. Speaker information
st.header("Speaker's sex,age")

sex_group = df_speaker['sex'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

for ax in axes:
    ax.margins(0.1,0.1)
    ax.spines[['top','right']].set_visible(False)

sex_bars=axes[0].bar(sex_group.index, sex_group, width=0.7,color='tomato')
axes[0].set_title('sex',fontsize=14)
axes[0].set_xlabel('sex',fontsize=14)
axes[0].set_ylabel('count',fontsize=14)
axes[0].set_xticks(range(len(sex_group.index)))    
axes[0].set_xticklabels(sex_group.index, fontsize=14)  # fontsize를 조정하여 글자 크기 설정


bins=[10,20,30,40,50,60,70]
age_bars=axes[1].hist(df_speaker['age'],bins,rwidth=0.9,linewidth=3.0)
axes[1].set_title('age',fontsize=14)
axes[1].set_xlabel("age",fontsize=14)


for i,bar in enumerate(sex_bars):
    width=bar.get_width()
    height=bar.get_height()
    x,y=bar.get_xy()
    sex_count=sex_group.iloc[i]
    axes[0].text(x+width/2,
             y+height*1.01,
             f'{sex_count}',
             ha='center',
             weight='bold')

st.pyplot(fig)


# 4. 대화마다 평균 발화 횟수, 평균 턴 횟수
st.header("utterances_info")
df_utterances['session_id']=(df_utterances['utterances_no']==1).cumsum()

session_turn_max=df_utterances.groupby('session_id')['turn'].max()
session_utterances_no_max=df_utterances.groupby('session_id')['utterances_no'].max()


turn_mean=session_turn_max.mean()
utterances_no_mean=session_utterances_no_max.mean()

fig,axes=plt.subplots(1,2,figsize=(15,7))
turn_count=session_turn_max.value_counts()
utterances_no_count=session_utterances_no_max.value_counts()
axes[0].bar(turn_count.index,turn_count)
axes[0].set_xlim(0,12)
axes[0].set_title("Turn counts",fontsize=14)
axes[0].set_xlabel("대화세션당 Turn 수",fontsize=14)
axes[0].set_ylabel("Counts",fontsize=14)

axes[1].bar(utterances_no_count.index,utterances_no_count)
axes[1].set_xlim(0,30)
axes[1].set_title("utterances counts",fontsize=14)
axes[1].set_xlabel("대화세션당 발화 수",fontsize=14)

st.pyplot(fig)

st.write(f"- turn 의 평균값 : {turn_mean:.2f}")
st.write(f"- utterances_no 의 평균값 : {utterances_no_mean:.2f}")
st.write(f"- total utterance session : {len(df_utterances_info)}")
