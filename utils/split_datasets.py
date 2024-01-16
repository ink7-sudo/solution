import os, gc, scipy.sparse
import pandas as pd
import numpy as np
import argparse
from sklearn.decomposition import TruncatedSVD
import config

parser = argparse.ArgumentParser(description='train')
parser.add_argument('-t', '--task', default='cite', type=str)
args = parser.parse_args()

subtask = args.task


def cite_test(data):
    metadata = pd.read_csv(DATA_DIR+'metadata.csv')
    metaday4 = metadata[metadata['day'] == 4]
    meta = metaday4[(metaday4['technology'] == 'citeseq' ) & ( metadata['donor']!=27678)] 
    merged_df = pd.merge(data, meta, on='cell_id')
    merged_df.drop(merged_df.columns[-4:], axis=1, inplace=True)
    merged_df.set_index('cell_id',inplace=True)
    return merged_df

def cite_train(data):
    metadata = pd.read_csv(DATA_DIR+'metadata.csv')
    metaday = metadata[((metadata['day'] == 2) | (metadata['day'] == 3) ) & ( metadata['donor']!=27678)]
    meta = metaday[metaday['technology'] == 'citeseq']
    merged_df = pd.merge(data,meta,on = 'cell_id')
    merged_df.drop(merged_df.columns[-4:], axis=1, inplace=True)
    merged_df.set_index('cell_id',inplace=True)
    return merged_df
    
if subtask == 'cite':
    DATA_DIR = os.path.join(config.RAW_DATA_DIR, "open-problems-multimodal/")
    PROCESSED_DATA_DIR = str(config.PROCESSED_DATA_DIR)
    FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")
    FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
    FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
    FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")
    print(PROCESSED_DATA_DIR)
    X = pd.read_hdf(FP_CITE_TRAIN_INPUTS)
    X_test = cite_test(X)
    X_train = cite_train(X)
    Y =  pd.read_hdf(FP_CITE_TRAIN_TARGETS)
    Y_test= cite_test(Y)
    Y_train = cite_train(Y)
    X_train.to_hdf(PROCESSED_DATA_DIR + '/cite_day23_train.h5',key = 'gene_id')
    Y_train.to_hdf(PROCESSED_DATA_DIR + '/cite_day23_target.h5',key = 'gene_id')
    X_test.to_hdf(PROCESSED_DATA_DIR + '/cite_day4_test.h5',key = 'gene_id')
    Y_test.to_hdf(PROCESSED_DATA_DIR + '/cite_day4_target.h5',key = 'gene_id')

    train_cite_targets_raw = pd.read_hdf(DATA_DIR+'train_cite_targets_raw.h5')
    train_cite_inputs_raw = pd.read_hdf(DATA_DIR+'train_cite_inputs_raw.h5')
    cite_day23_row  = train_cite_targets_raw.iloc[:42843, :]
    cite_day23_row.to_hdf(PROCESSED_DATA_DIR + '/train_cite_day23_targets_raw.h5',key='geneid')
    cite_day23_row = train_cite_inputs_raw.iloc[:42843, :]
    cite_day4_row = train_cite_inputs_raw.iloc[42843:, :]
    cite_day23_row.to_hdf(PROCESSED_DATA_DIR+ '/train_cite_day23_inputs_raw.h5',key='geneid')
    cite_day4_row.to_hdf(PROCESSED_DATA_DIR + '/test_cite_day4_inputs_raw.h5',key='geneid')


