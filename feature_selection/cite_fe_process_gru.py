import gc
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import mudata
from mudata import AnnData, MuData
import scanpy as sc
import muon as mu
from scipy.sparse import hstack,vstack,csr_matrix,save_npz,load_npz
from sklearn.decomposition import NMF,LatentDirichletAllocation,TruncatedSVD
from muon import prot as pt
import json

############################################################################
#----- work folder -----
############################################################################
settings ={"input_path":"/home/dujunjia/bio/solution/data/open-problems-multimodal/",
            "features_path":"/home/dujunjia/bio/solution/result/cite_fe_gru/"}

input_path = settings['input_path']
feature_path = settings['features_path']

# save transformed cite inputs to csr_matrix and id list
train_cite_inputs = pd.read_hdf(input_path+'cite_day23_train.h5').reset_index(drop=True)
metadata = pd.read_csv(input_path+'metadata.csv')


# id list
train_cite_inputs_id = pd.read_hdf(input_path+'cite_day23_train.h5').reset_index()[['cell_id']]
train_cite_inputs_id = train_cite_inputs_id.merge(metadata,on=['cell_id'],how='left')
train_cite_inputs_id.to_feather(feature_path+'train_cite_inputs_id.feather')

# csr_matrix
train_cite_inputs_sparse = csr_matrix(train_cite_inputs.to_numpy())
save_npz(feature_path+"day4_train_cite_inputs_sparse.npz", train_cite_inputs_sparse)



test_cite_inputs = pd.read_hdf(input_path+'cite_day4_test.h5').reset_index(drop=True)

# id list
test_cite_inputs_id = pd.read_hdf(input_path+'cite_day4_test.h5').reset_index()[['cell_id']]
test_cite_inputs_id = test_cite_inputs_id.merge(metadata,on=['cell_id'],how='left')
test_cite_inputs_id.to_feather(feature_path+'test_cite_inputs_id.feather')

# csr_matrix
test_cite_inputs_sparse = csr_matrix(test_cite_inputs.to_numpy())
save_npz(feature_path+"day4_test_cite_inputs_sparse.npz", test_cite_inputs_sparse)


cite_inputs_sparse = vstack([train_cite_inputs_sparse,test_cite_inputs_sparse])
save_npz(feature_path+"day4_cite_inputs_sparse.npz", cite_inputs_sparse)


# cite_inputs_sparse
del train_cite_inputs,train_cite_inputs_sparse,test_cite_inputs,test_cite_inputs_sparse,cite_inputs_sparse
gc.collect()


# save raw count cite inputs to csr_matrix 
train_cite_inputs_raw = pd.read_hdf(input_path+'train_cite_day23_inputs_raw.h5').reset_index(drop=True)
train_cite_inputs_raw_sparse = csr_matrix(train_cite_inputs_raw.to_numpy())

test_cite_inputs_raw = pd.read_hdf(input_path+'test_cite_day4_inputs_raw.h5').reset_index(drop=True)
test_cite_inputs_raw_sparse = csr_matrix(test_cite_inputs_raw.to_numpy())

cite_inputs_raw_sparse = vstack([train_cite_inputs_raw_sparse,test_cite_inputs_raw_sparse])
save_npz(feature_path+"day4_cite_inputs_raw_sparse.npz", cite_inputs_raw_sparse)


# cite_inputs_raw_sparse
del train_cite_inputs_raw,train_cite_inputs_raw_sparse,test_cite_inputs_raw,test_cite_inputs_raw_sparse,cite_inputs_raw_sparse
gc.collect()


# # save target to numpy
train_cite_targets = pd.read_hdf(input_path+'cite_day23_target.h5').reset_index(drop=True)
np.save(feature_path+'day4_train_cite_targets.npy', train_cite_targets)


del train_cite_targets
gc.collect()


# # centered log ratio(clr) for raw count
train_rna_df = pd.read_hdf(input_path+'train_cite_day23_inputs_raw.h5')
test_rna_df = pd.read_hdf(input_path+'test_cite_day4_inputs_raw.h5')
rna_df = pd.concat([train_rna_df,test_rna_df])
rna = AnnData(csr_matrix(rna_df))
rna.obs_names = rna_df.index.values  
rna.var_names = rna_df.columns.values


pt.pp.clr(rna)

cite_inputs_clr_sparse = rna.X
save_npz(feature_path+'day4_cite_inputs_clr_sparse.npz', cite_inputs_clr_sparse)


del train_rna_df,test_rna_df,rna_df,rna,cite_inputs_clr_sparse
gc.collect()


# # clr to 200d tsvd
cite_inputs_clr_sparse = load_npz(feature_path+"day4_cite_inputs_clr_sparse.npz")
print ('cite_inputs_clr_sparse',cite_inputs_clr_sparse.shape)
tsvd = TruncatedSVD(n_components=200, algorithm='arpack')
cite_inputs_svd = tsvd.fit_transform(cite_inputs_clr_sparse)
np.save(feature_path+'day4_cite_inputs_svd_clr_200.npy', cite_inputs_svd)


del cite_inputs_clr_sparse,tsvd,cite_inputs_svd
gc.collect()


# fine tuned processV
# load inputs merge with metadata


print('load raw')
train_cite_inputs_raw = pd.read_hdf(input_path+'train_cite_day23_inputs_raw.h5').reset_index()
test_cite_inputs_raw = pd.read_hdf(input_path+'test_cite_day4_inputs_raw.h5').reset_index()
cite_inputs_raw = pd.concat([train_cite_inputs_raw, test_cite_inputs_raw]).reset_index(drop=True)

del train_cite_inputs_raw,test_cite_inputs_raw
gc.collect()

print('load metadata')
metadata = pd.read_csv(input_path+'metadata.csv')
cite_inputs_raw = cite_inputs_raw.merge(metadata, on=['cell_id'], how='left')


#  divide by mean then sqrt transformation



print('start process')
# constant_cols copy from https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart


with open('constant_cols_raw.txt', 'r') as file:
    # 使用strip()去除每行末尾的换行符，并将每行转换为整数
    constant_cols = [line.strip() for line in file]



use_cols = [f for f in cite_inputs_raw.columns if f not in constant_cols if f not in ['cell_id','day','donor','cell_type','technology']]
cite_inputs_raw_array = cite_inputs_raw[use_cols].values

cite_inputs_raw_row_mean = cite_inputs_raw_array.mean(axis=1).reshape(-1, 1)
cite_inputs_raw_norm = cite_inputs_raw_array / cite_inputs_raw_row_mean

cite_inputs_raw_norm = np.sqrt(cite_inputs_raw_norm)



# zscore by column

cite_inputs_normalization = np.zeros((cite_inputs_raw_norm.shape[0], cite_inputs_raw_norm.shape[1]))
    
for j in range(cite_inputs_raw_norm.shape[1]):
    x_col = cite_inputs_raw_norm[:,j]
    cite_inputs_normalization[:,j] = (x_col - np.mean(x_col)) / np.std(x_col)    


# batch-effect correction


cite_inputs_raw_day = cite_inputs_raw[['day']]
cite_inputs_raw_day = pd.concat([cite_inputs_raw_day, pd.DataFrame(cite_inputs_normalization)],axis=1)
cite_inputs_raw_day


cite_inputs_raw_day_median = cite_inputs_raw_day.groupby(['day']).transform('median').values
cite_inputs_raw_day_median


cite_inputs_final_normalization = cite_inputs_normalization - cite_inputs_raw_day_median


del cite_inputs_normalization,cite_inputs_raw_day_median,cite_inputs_raw_day
gc.collect()


np.save(feature_path+'day4_cite_inputs_final_normalization.npy', cite_inputs_final_normalization)


# ### dimession reduction
print ('svd')
cite_inputs_bio_norm = np.load(feature_path+'day4_cite_inputs_final_normalization.npy')
tsvd = TruncatedSVD(n_components=100, algorithm='arpack')
cite_inputs_bio_norm_svd = tsvd.fit_transform(cite_inputs_bio_norm)
np.save(feature_path+'day4_cite_inputs_bio_norm_svd_100.npy', cite_inputs_bio_norm_svd)


from sklearn.decomposition import *   
cite_inputs_bio_norm = np.load(feature_path+'day4_cite_inputs_final_normalization.npy')

######################## PCA ########################
print ('PCA')
pca = PCA(n_components = 64, 
          copy = False, )
cite_inputs_bio_norm_pca = pca.fit_transform(cite_inputs_bio_norm)
np.save(feature_path+'day4_cite_inputs_bio_norm_pca_64.npy', cite_inputs_bio_norm_pca)



# # correlated and important features selection
print('load raw')
train_cite_inputs_raw = pd.read_hdf(input_path+'train_cite_day23_inputs_raw.h5').reset_index()

print('load metadata')
metadata = pd.read_csv(input_path+'metadata.csv')
train_cite_inputs_raw = train_cite_inputs_raw.merge(metadata, on=['cell_id'], how='left')

del metadata
gc.collect()


train_cite_inputs_raw['grp'] = train_cite_inputs_raw['donor'].astype('str')+'_'+train_cite_inputs_raw['day'].astype('str')
grp_all = train_cite_inputs_raw['grp'].values
grp_u = sorted(set(grp_all))
grp_u


train_raw = pd.read_hdf(input_path+'cite_day23_train.h5').astype(np.float16)
cols_cite = train_raw.columns
train_raw = train_raw.values

train_cite_target = pd.read_hdf(input_path+'cite_day23_target.h5').astype(np.float16)
labels_cite = train_cite_target.columns
train_cite_target = train_cite_target.values
train_cite_target -= train_cite_target.mean(axis=1).reshape(-1, 1)
train_cite_target /= train_cite_target.astype('float32').std(axis=1).reshape(-1, 1)
train_cite_target = train_cite_target.astype('float16')


from tqdm import tqdm 
#### corr ####
cor_LIST = {}
for grp in grp_u:
    con = (grp_all == grp)
    X_p = train_raw[con]
    y_p = train_cite_target[con]

    chunksize = 1000
    chunk = range(0,X_p.shape[1],chunksize)
    cor_df = []
    for i in tqdm(chunk):
        s = i
        e = i+chunksize
        if i == chunk[-1]:
            e = X_p.shape[1]
        cor_df_i = np.corrcoef(X_p[:,s:e].T,y_p.T)
        cor_df_i = cor_df_i[:-140,-140:]
        cor_df.append(cor_df_i)
    cor_df = np.concatenate(cor_df,0)
    cor_LIST[grp] = cor_df

    del X_p,y_p
    gc.collect()

cor_Min = {}
for col in labels_cite:
    cor_Min[col] = []

for grp in cor_LIST:
    cor_df = pd.DataFrame(cor_LIST[grp],index = cols_cite,columns=labels_cite)
    for col in cor_df.columns:
        cor_Min[col].append(cor_df[[col]])

for col in tqdm(labels_cite):
    cor_Min[col] = pd.concat(cor_Min[col],1).dropna()
    cor_Min[col]['min'] = cor_Min[col].quantile(axis = 1,q = 0.1).astype('float16')

sele_n = 10
cor_feats = []
for col in cor_Min:
    feat = cor_Min[col].sort_values('min',ascending=False).index.tolist()[:sele_n]
    for f in feat:
        cor_feats.append(f)
    
cor_feats = sorted(set(cor_feats))

# reference important_cols from https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart
with open('/home/dujunjia/bio/solution/code/important_cols.txt', 'r') as file:
  
    knowl_feats = [line.strip() for line in file]

important_feats = sorted(set(cor_feats+knowl_feats))
print(important_feats)
cite_inputs_raw_important_feats = cite_inputs_raw[important_feats].values
np.save(feature_path+'day4_cite_inputs_raw_important_feats.npy', cite_inputs_raw_important_feats)