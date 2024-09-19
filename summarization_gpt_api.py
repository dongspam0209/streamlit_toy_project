from tqdm import tqdm
import pandas as pd
from openai import OpenAI

client = OpenAI() # 컴터 환경변수에 api key 저장해둠


input_csv="(toy)_sampling_data_20240825.csv"
output_csv="(toy)_sampling_data_summarization_result_0828.csv"
column_to_summarize='utterances_text'

dataset=pd.read_csv(input_csv)
dataset.dropna(inplace=True)
print(dataset)

def summarization_text(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # gpt-4 model 사용
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a summarization assistant that specializes in processing multi-turn conversations in Korean. "
                        "Your task is to summarize the provided conversation into at most two sentence."
                        "Whenever possible, try to use only the words and information present in the original text."
                        "Avoid introducing new concepts or words that are not found in the text unless absolutely necessary."
                        )
                },

                {
                    "role": "user", 
                    "content": f"Please summarize the following multi-turn conversation into two sentences at most. Whenever possible, use only the words from the conversation text. Conversation: {text}"
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=0.9
        )
        summarization_result = response.choices[0].message.content # 응답 indexing
        return summarization_result
    
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None


for idx,row in tqdm(dataset.iterrows(), total=len(dataset)): # dataset에서 column_to_summarize('utterances_text')에 해당하는 text만 summarize 요청
    try:
        summarized_text=summarization_text(row[column_to_summarize])
        if summarized_text:  # 요약이 성공했을 경우에만 저장
            dataset.at[idx, 'utterances_summarize_result'] = summarized_text # 새로운 utterances_summarize_result column에 요약된 문자열 저장
    except Exception as e:
        print(f"Error in row {idx}: {e}")


    if idx%10==0:
        dataset.to_csv(output_csv,index=False) # api credit 다 써버려서 튕길 수 있으니 10개의 sample마다 중간중간에 저장
        print(f'saved at {idx}')


dataset.to_csv(output_csv,index=False,encoding='utf-8-sig') # 최종 요약결과 저장된 dataframe 추출
print("complete")

