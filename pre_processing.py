import os
import json
import traceback
import pandas as pd
from tqdm import tqdm


PATH=r'./012.한국어 SNS 멀티턴 대화 데이터/3.개방데이터/1.데이터'
train_label_path=os.path.join(PATH,'Training/02.라벨링데이터')
valid_label_path=os.path.join(PATH,'Validation/02.라벨링데이터')
          

def get_dir_list(src_path):
    dir_ls = []
    """ Raw dataset에서 입력한 인자의 하위 디렉토리를 반환하는 함수

    Returns:
        list : 각 element는 디렉토리 경로
    """    
    for path, dirs, files in os.walk(src_path):
        for dir_name in dirs:
            current_dir = os.path.join(path, dir_name).replace('\\', '/')
            dir_ls.append(current_dir)

    return dir_ls

def json_to_dataframe(json_file, is_train, topic_to_id_map, topic_id):
    """
    get_file_list로 생성한 디렉토리 내에 존재하는 json 형식의 파일로부터, dataframe을 생성하는 함수

    Args:
        json_file (str): 변환할 json 파일명
        topic_id (int): get_file_list로 생성한 디렉토리의 번호에 해당하며, 각 토픽에 대한 번호
        is_train (int): 해당 json이 원천데이터에 있었으면 1, 라벨링데이터에 있었으면 0으로 입력됨
        topic_to_id_map(dict): 이미 있는 topic이면 넘어가고 topic_id를 증가해서 부여한다.
    """    
    with open(json_file,'r',encoding='utf-8') as f:
        data=json.load(f) # json module 사용해서 load
        topic=data['info']['topic']

        if is_train==1:
            if topic not in topic_to_id_map:
                topic_to_id_map[topic]=topic_id
                topic_id+=1
            current_topic_id=topic_to_id_map[topic]
        else:
            current_topic_id=topic_to_id_map.get(topic,None)
            if current_topic_id is None:
                raise ValueError(f"Validation 데이터에서 '{topic}'에 대한 topic_id를 찾을 수 없습니다.")
        
    
    """
    Topic table
    """
    topic_data={
        'id': current_topic_id, # 디렉토리 종류에 따라 id 부여 topic에 따른 id 9가지
        'topic': [data['info']['topic']]
    }
    topic_df=pd.DataFrame(topic_data)
    topic_df=topic_df.astype({'id':int,'topic':str})

    """
    Keyword table
    """
    keyword_data={
        'id':[data['info']['id']], # 대화에 따른 id, dataset의 크기만큼 존재
        'topic_id':current_topic_id,
        'keyword':[data['info']['keyword']]
    }
    keyword_df=pd.DataFrame(keyword_data)
    keyword_df.astype({'id':int,'topic_id':int,'keyword':str})

    """
    Speaker table
    """
    speakers=[]
    speaker_info=data['info']['speaker'] # speaker : {A/B/C ID/Sex/Age} 총 9가지 존재

    # Speaker A 처리
    speakers.append({
        'id': speaker_info.get('speakerAId', None),
        'sex': speaker_info.get('speakerASex', None),
        'age': speaker_info.get('speakerAAge', None)
    })

    # Speaker B 처리
    speakers.append({   
        'id': speaker_info.get('speakerBId', None),
        'sex': speaker_info.get('speakerBSex', None),
        'age': speaker_info.get('speakerBAge', None)
    })

    # Speaker C 처리
    speakers.append({   
        'id': speaker_info.get('speakerCId', None),
        'sex': speaker_info.get('speakerCSex', None),
        'age': speaker_info.get('speakerCAge', None)
    })

    speaker_df=pd.DataFrame(speakers)
    speaker_df['id']=speaker_df['id'].fillna(0)
    speaker_df['age']=speaker_df['age'].fillna(0)
    speaker_df=speaker_df.astype({'id':int,'sex':str,'age':int})
    
    """
    Utterances_info table
    """
    utterances_info_data={
        'keyword_id':[data['info']['id']], # 대화 id
        'topic_id':current_topic_id, # topic id
        'speaker_a_id': [speaker_df.loc[0, 'speaker_id']],
        'speaker_b_id': [speaker_df.loc[1, 'speaker_id']],
        'speaker_c_id': [speaker_df.loc[2, 'speaker_id']],
        'is_train':is_train
    }
    utterances_info_df = pd.DataFrame(utterances_info_data)
    utterances_info_df=utterances_info_df.astype({'keyword_id':int,'topic_id':int,'speaker_a_id':int,'speaker_b_id':int,'speaker_c_id':int,'is_train':int})
    """
    Utterances table
    """
    utterances = data['utterances']
    utterances_data = []
    for utterance in utterances:
        speaker_id = (speaker_df.loc[0, 'id'] if utterance['speaker'] == 'speakerA' 
                      else speaker_df.loc[1, 'id'] if utterance['speaker'] == 'speakerB' 
                      else speaker_df.loc[2, 'id'])
        
        utterances_data.append({
            'keyword_id': data['info']['id'],
            'turn': int(utterance['turn_id'].split('-')[-1]),
            'speaker_id': speaker_id,
            'utterances': utterance['text'],
            'utterances_no': int(utterance['utterance_id'].split('.')[-1]),
            'new_word':utterance['new_word'],
            'speech_act':utterance['speech_act']
        })

    utterances_df = pd.DataFrame(utterances_data)
    utterances_df=utterances_df.astype({'keyword_id':int,'turn':int,'speaker_id':int,'utterances':str,
                                        'utterances_no':int,'new_word':str,'speech_act':str})

    return topic_df, keyword_df, speaker_df, utterances_info_df, utterances_df, topic_id


def process_files_in_dir(dir_list, is_train, topic_to_id_map, topic_id):
    """
    디렉토리 내에 있는 json 파일을 dataframe으로 반환하기 위해서 json_to_dataframe 함수를 실행하는 함수

    Args:
        dir_list (str): train,valid directory list명
        is_train (bool): train,valid set을 구별하기 위한 data
        topic_to_id_map (dict): 이미 있는 topic이면 넘어가고 topic_id를 증가해서 부여한다.
        topic_id (int): topic의 id번호

    Returns:
        dataframes: topic,keyword,speaker,utterances_info,utterances 테이블의 dataframe을 반환한다.
    """    
    topic_df_list=[]
    keyword_df_list=[]
    speaker_df_list = []
    utterances_info_df_list=[]
    utterances_df_list = []

    for topic_dir in dir_list:
        for json_file in tqdm(os.listdir(topic_dir),desc="processing directories"):
            if json_file.endswith('.json'):
                json_path = os.path.join(topic_dir, json_file)
                topic_df,keyword_df,speaker_df, utterances_info_df, utterances_df, topic_id = json_to_dataframe(json_path, is_train, topic_to_id_map, topic_id)
                

                if not topic_df.empty:
                    topic_df_list.append(topic_df)                    
                if not keyword_df.empty:
                    keyword_df_list.append(keyword_df)
                if not speaker_df.empty:
                    speaker_df_list.append(speaker_df)
                if not utterances_info_df.empty:
                    utterances_info_df_list.append(utterances_info_df)
                if not utterances_df.empty:
                    utterances_df_list.append(utterances_df)


    return pd.concat(topic_df_list, ignore_index=True),pd.concat(keyword_df_list, ignore_index=True),pd.concat(speaker_df_list, ignore_index=True),pd.concat(utterances_info_df_list, ignore_index=True), pd.concat(utterances_df_list, ignore_index=True), topic_id


if __name__=="__main__":
    train_dirs=get_dir_list(train_label_path) # 훈련 데이터에 해당하는 디렉토리 경로 리스트
    valid_dirs=get_dir_list(valid_label_path) # 검증 데이터에 해당하는 디렉토리 경로 리스트

    topic_to_id_map={}
    topic_id = 1  # Topic ID 시작

     # Training 데이터 처리
    final_topic_df_train,final_keyword_df_train,final_speaker_df_train,final_utterances_info_df_train, final_utterances_df_train, topic_id = process_files_in_dir(train_dirs, 1, topic_to_id_map, topic_id)

    # Validation 데이터 처리
    final_topic_df_valid,final_keyword_df_valid,final_speaker_df_valid,final_utterances_info_df_valid, final_utterances_df_valid, _ = process_files_in_dir(valid_dirs, 0, topic_to_id_map, topic_id)

    # 최종 데이터프레임 결합
    final_topic_df = pd.concat([final_topic_df_train, final_topic_df_valid], ignore_index=True)
    final_keyword_df = pd.concat([final_keyword_df_train, final_keyword_df_valid], ignore_index=True)
    final_speaker_df = pd.concat([final_speaker_df_train, final_speaker_df_valid], ignore_index=True)
    final_utterances_info_df = pd.concat([final_utterances_info_df_train, final_utterances_info_df_valid], ignore_index=True)
    final_utterances_df = pd.concat([final_utterances_df_train, final_utterances_df_valid], ignore_index=True)

    # 최종 데이터프레임 확인
    print("Final Topic DataFrame:\n", final_topic_df.head())
    print("Final Keyword DataFrame:\n", final_keyword_df.head())
    print("Final Speaker DataFrame:\n", final_speaker_df.head())
    print("Final Utterances info DataFrame:\n", final_utterances_info_df.head())
    print("Final Utterances DataFrame:\n", final_utterances_df.head())

    # dataset csv로 저장
    final_topic_df.to_csv("topic_table.csv", encoding='utf-8-sig', index=False)
    final_keyword_df.to_csv("keyword_table.csv", encoding='utf-8-sig', index=False)
    final_speaker_df.to_csv("speaker_table.csv", encoding='utf-8-sig', index=False)
    final_utterances_info_df.to_csv("utterances_info_table.csv", encoding='utf-8-sig', index=False)
    final_utterances_df.to_csv("utterances_table.csv", encoding='utf-8-sig', index=False)
