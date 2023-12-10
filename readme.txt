运行步骤
1 下载数据集 （https://www.kaggle.com/competitions/open-problems-multimodal/data）到data文件夹 open-problems-multimodal 路径下
   下载数据集（https://www.kaggle.com/datasets/fabiencrom/multimodal-single-cell-as-sparse-matrix）到data文件夹 multimodal-single-cell-as-sparse-matrix路径下
2 运行code/fe_process_cite_lgb.py文件，该步骤主要目的是生成cite任务lgb模型的特征文件，运行需要一定时间
3 运行code/cite_lgb.py，这个代码训练lgb预测 cite 任务
4 运行code/cite_fe_process_gru.py文件，
        cite_fe_process_lgb1_meta_feature.py，
        cite_fe_process_lgb2_meta_feature.py，
        cite_fe_process_lgb3_meta_feature.py，
        cite_fe_process_lgb4_meta_feature.py，该步骤主要目的是生成cite任务gru模型的特征文件，运行需要一定时间
5 运行code/cite_gru.py，这个代码训练gru预测 cite 任务
6 运行code/fe_process_multi.py文件，该步骤主要目的是生成multi任务的特征文件，运行需要一定时间
7 运行code/multi_mlp.py，这个代码训练mlp预测multi任务
5 运行code/infer.ipynb，得到两个任务的拼接结果
7 提交result/sub/submission.csv文件到kaggle


一些依赖的包需要自行安装一下

pyarrow
torch
pickle
tables
pandas==1.3.5
numpy==1.19.5
scipy==1.6.3
sklearn==0.24.2
joblib==1.0.1
lightgbm==3.2.1
tqdm==4.61.0
tensorflow==2.9.2
tensorflow_addons==0.18.0
运行需要内存较大，建议64G内存，运行时间较长





