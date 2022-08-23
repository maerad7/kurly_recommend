from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
# csv 로딩
df = pd.read_csv("./not_null_df.csv")
# 데이터 전처리
df = df.drop(['Unnamed: 0'],axis=1)
del_index = df[df.skin_info=="복합성"].index
df = df.drop(index=del_index,axis=0)

# 상품 명과 유저 아이디를 id 값으로 변환
user_to_replace_id = {original: idx for idx, original in enumerate(sorted(df.user_id.unique()))}
item_to_replace_id = {original: idx for idx, original in enumerate(sorted(df.item_id.unique()))}
df['item_to_replace_id'] = df["item_id"].map(lambda x:item_to_replace_id[x])
df['user_replace_id'] = df['user_id'].map(lambda x:user_to_replace_id[x])

# 피부 정보들
skin_type_dict = { "1": '건성' , "2" :'지성', "3":'중성', '4':'수분부족지성', "5":'복합성', "6":'극건성'}
skin_info_dict = { "1":'민감성', "2" :'모공', "3":'탄력없음', "4":'칙칙함', '5':'트러블', "6":'건조함', "7":'주름'}


# 필요한 칼럼만 뽑아서 result_df 에 저장
result_df = df[['user_replace_id',"item_to_replace_id","user_rating","skin_type","skin_info","item_id","user_id","user_rating"]]
ratings = result_df

# id to index
item_to_index_dict = {}
for item_name,item_id in zip(ratings.item_id.unique(),ratings.item_to_replace_id.unique()):
    item_to_index_dict[item_name] = item_id

# index to id
index_to_item_id_dict = {}
for item_name,item_id in zip(ratings.item_id.unique(),ratings.item_to_replace_id.unique()):
    index_to_item_id_dict[item_id] = item_name

def Item_base_cos_df(ratings):
    item_based = ratings.pivot_table('user_rating',index="item_to_replace_id",columns="user_replace_id")
    item_based.fillna(0,inplace=True)
    sim_rate = cosine_similarity(item_based,item_based)
    sim_rate_df = pd.DataFrame(
        data = sim_rate,
        index= item_based.index,
        columns = item_based.index
    )
    return sim_rate_df

def recommend(sim_rate_df,item_id,k):
    return sim_rate_df[item_id].sort_values(ascending=False)[1:k+1]

def index_to_item_id(result_list):
    item_ids = []
    for result in result_list:
        item_id = index_to_item_id_dict[result]
        item_ids.append(int(item_id))
    return item_ids

def Item_based_filtering(df,parmas_skin_type,params_skin_info,item_name,recommend_data_nums):
    item_id = item_to_index_dict[item_name] 
    skin_type=df[df["skin_type"]==parmas_skin_type]
    skin_info=skin_type[skin_type['skin_info']==params_skin_info]
    result = Item_base_cos_df(skin_info)
    result_list = recommend(result,item_id,recommend_data_nums)
    item_id_list = index_to_item_id(result_list.index)
    return item_id_list

@app.route('/predict', methods=['GET','POST']) 
def predict():
    json_request=None
    if request.method == "POST":
        data = request.json
        user_skin_type = data['skin_type']
        user_skin_info = data['skin_info']
        print(user_skin_type,user_skin_type)

        item_name = data['item_name']
        recommend_data_nums = data['recommend_data_nums']
        user_skin_type = skin_type_dict[user_skin_type]
        user_skin_info = skin_info_dict[user_skin_info]
        print(user_skin_type,user_skin_info)
        prediction = Item_based_filtering(df,user_skin_type,user_skin_info,item_name,recommend_data_nums)
        send_data = {"data": prediction}
        
    return jsonify(send_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)