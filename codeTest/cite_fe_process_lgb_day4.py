import os, gc, scipy.sparse
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD

DATA_DIR = "/home/dujunjia/bio/solution/data/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"cite_day23_train.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"cite_day23_target.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"cite_day4_test.h5")

print(FP_CITE_TEST_INPUTS)

with open('/home/dujunjia/bio/solution/code/constant_cols.txt', 'r') as file:
    # 使用strip()去除每行末尾的换行符，并将每行转换为整数
    constant_cols = [line.strip() for line in file]


with open('/home/dujunjia/bio/solution/code/important_cols.txt', 'r') as file:
    # 使用strip()去除每行末尾的换行符，并将每行转换为整数
    important_cols = [line.strip() for line in file]


X = pd.read_hdf(FP_CITE_TRAIN_INPUTS).drop(columns=constant_cols)
X0 = X[important_cols].values
print(f"Original X shape: {str(X.shape):14} {X.size*4/1024/1024/1024:2.3f} GByte")
gc.collect()
X = scipy.sparse.csr_matrix(X.values)
gc.collect()

Xt = pd.read_hdf(FP_CITE_TEST_INPUTS).drop(columns=constant_cols)
X0t = Xt[important_cols].values
print(f"Original Xt shape: {str(Xt.shape):14} {Xt.size*4/1024/1024/1024:2.3f} GByte")
gc.collect()
Xt = scipy.sparse.csr_matrix(Xt.values)


both = scipy.sparse.vstack([X, Xt])
assert both.shape[0] == 70988
print(f"Shape of both before SVD: {both.shape}")
svd = TruncatedSVD(n_components=512, random_state=1) 
both = svd.fit_transform(both)
print(f"Shape of both after SVD:  {both.shape}")


# Hstack the svd output with the important features
X = both[:42843]
Xt = both[42843:]
del both
X = np.hstack([X, X0])
Xt = np.hstack([Xt, X0t])
print(f"Reduced X shape:  {str(X.shape):14} {X.size*4/1024/1024/1024:2.3f} GByte")
print(f"Reduced Xt shape: {str(Xt.shape):14} {Xt.size*4/1024/1024/1024:2.3f} GByte")

pd.DataFrame(X).to_csv('/home/dujunjia/bio/solution/result/cite_fe_lgb/X_876_day4.csv', index=False)
pd.DataFrame(Xt).to_csv('/home/dujunjia/bio/solution/result/cite_fe_lgb/Xt_876_day4.csv', index=False)

