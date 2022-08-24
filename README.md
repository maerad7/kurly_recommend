# kurly_recommend
해커톤 추천시스템
# package
- numpy
- pandas
- scikit-learn
- tensorflow
- wandb

# Comment
- 학습코드는 colab 에서 실행했습니다. ipynb 형태입니다.
- 인퍼런스 서버 파일명은 cosmetic_item_cf_filtering_server.py
- datamerge.ipynb 를 실행해 all_df.csv 생성
- deep_learning.ipynb 는 딥러닝 학습 코드입니다. 맨 마지막에 모델파일 experiment_no_onehot_add_category_05.h5를 생성합니다.
- Item_base_CF.ipynb 는 Item based collaborative filltering 예제 코드입니다.
- all_df.csv 는 전체 데이터셋 입니다.
- not_null_df.scv 는 학습에 사용할 데이터 셋 입니다.
- last_dataset.csv 는 데이터 전처리를 끝난 데이터 셋 입니다. 

# 실행 필요 파일
- cosmetic_item_cf_filtering_server.py, experiment_no_onehot_add_category_05.h5, not_null_df.scv
- 위 3개 파일만 있으면 학습서버 띄워서 인퍼런스 가능합니다.
- 모델 파일 만들기 위한 deep_learning.ipynb 
- 데이터 셋인 all_df.csv 
