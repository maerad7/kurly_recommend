from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

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
skin_type_dict = {0:'건성'  , 1:'지성', 2:'중성', 3:'수분부족지성', 4:'복합성', 5:'극건성'}
skin_info_dict = {0:'민감성', 1:'모공', 2:'탄력없음', 3:'칙칙함', 4:'트러블', 5:'건조함', 6:'주름'}


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

# deep_learning =====================

# 아이템 전체 리스트
item_list = df.item_to_replace_id.unique()

# 딥러닝 모델 로드
model = tf.keras.models.load_model('experiment_no_onehot_add_category_05.h5')

def index_to_item_id(result_list):
    item_ids = []
    for result in result_list:
        item_id = index_to_item_id_dict[result]
        item_ids.append(int(item_id))
    return item_ids

# 딥러닝 모델 호출
def model_predict(recommendation_model, user_id,item_id,skin_type,skin_info,age_group,gender,category,recommend_data_nums):
    user = [user_to_replace_id[user_id]] *len(item_list)
    skin_type = [skin_type]*len(item_list)
    skin_info = [skin_info]*len(item_list)
    age_group = [age_group]*len(item_list)
    # gender = 1 if gender == "남성" else 0
    gender = [gender]*len(item_list)
    category =[category]*len(item_list)
    user = np.array(user)
    skin_type = np.array(skin_type)
    skin_info = np.array(skin_info)
    age_group = np.array(age_group)
    gender = np.array(gender)
    category = np.array(category)

    prediction = recommendation_model.predict([user,item_id,skin_type,skin_info,age_group,gender,category])
    result_list = get_recommendation_result(recommend_data_nums,prediction)
    int_result_list = []
    for i in result_list:
        int_result_list.append(int(i))
    result = index_to_item_id(int_result_list)     
    res_data = {"recommend_list" : result }
    return res_data


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



def Item_based_filtering(df,parmas_skin_type,params_skin_info,item_name,recommend_data_nums):
    item_id = item_to_index_dict[item_name] 
    skin_type=df[df["skin_type"]==parmas_skin_type]
    skin_info=skin_type[skin_type['skin_info']==params_skin_info]
    result = Item_base_cos_df(skin_info)
    result_list = recommend(result,item_id,recommend_data_nums)
    item_id_list = index_to_item_id(result_list.index)
    return item_id_list

def get_recommendation_result(k, prediction):
    result = tf.nn.top_k(prediction.reshape(397), k=k).indices
    result = list(result.numpy())
    return result

@app.route('/predict', methods=['GET','POST']) 
def predict():
    json_request=None
    send_data={}
    if request.method == "POST":
        try:
            data = request.json
            user_skin_type = data['skin_type']
            user_skin_info = data['skin_info']
            print(user_skin_type,user_skin_type)

            item_name = data['item_name']
            recommend_data_nums = data['recommend_data_nums']
            user_skin_type = skin_type_dict[user_skin_type]
            user_skin_info = skin_info_dict[user_skin_info]
            # print(user_skin_type,user_skin_info)
            prediction = Item_based_filtering(df,user_skin_type,user_skin_info,item_name,recommend_data_nums)
            send_data = {"recommend_list": prediction}
        except:
            # 추천 리스트가 없습니다.
            send_data = {"recommend_list": None}
    return jsonify(send_data)

@app.route('/deep_learning_predict', methods=['GET','POST']) 
def deep_learning_predict():
    json_request=None
    send_data={}
    
    if request.method == "POST":
        try:   
            data = request.json
            user_id = data['user_id']
            skin_type = data['skin_type']
            skin_info = data['skin_info']
            age_group = data['age_group']/10 
            gender = data['gender'] # 남성 1 여성 0
            category = data['category']
            recommend_data_nums = data['recommend_data_nums']

            send_data = model_predict(model,user_id,item_list,skin_type,skin_info,age_group,gender,category,recommend_data_nums)
        except:
            # 추천 리스트가 없습니다.
            send_data = {"recommend_list": None}
    return jsonify(send_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)