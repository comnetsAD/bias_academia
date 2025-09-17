import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import pandas as pd
import numpy as np
import glob
import sys
IND = int(sys.argv[1])

logging.info(IND)

model = gensim.models.Doc2Vec.load('/scratch/fl1092/OpenAlex/openalex/derived_data/abstract_doc2vec.model')

logging.info(model.corpus_count) # 14263546 if 10%, 42790656 if 30%


OPENALEX = '/scratch/fl1092/OpenAlex/openalex'

def printShape(df, cols=[], msg=''):
    
    print(df.shape, end='  ')
    for col in cols:
        print(col, df[col].nunique(), end='  ')
    print(msg, flush=True)
    
    return df

def inferVector(row):

    doc = row['Abstract']
    ID = row['PaperID']

    try:
        return model.dv[ID]
    except Exception as e:
    
        tokens = gensim.utils.simple_preprocess(doc)
        
        return model.infer_vector(tokens)



inFile = f'{OPENALEX}/cleaned_data/PaperAbstract/{IND}.csv'
outCSV = f'{OPENALEX}/derived_data/abstract_doc2vec_embeddings/{IND}_PaperIDs.csv'
outNPY = f'{OPENALEX}/derived_data/abstract_doc2vec_embeddings/{IND}_embeddings.npy'

if len(glob.glob(outNPY)) != 0:
    raise Exception('File already exists!')

df = pd.read_csv(inFile).dropna().pipe(printShape)
df.to_csv(outCSV, columns=['PaperID'], index=False)

logging.info(df.shape)

df = df.assign(PaperID=lambda df: df.PaperID.astype(str))
df = df.assign(Embeddings=lambda df: df.apply(inferVector, axis=1))

logging.info('Embeddings done')

with open(outNPY, 'wb') as f:
    np.save(f, df.Embeddings)