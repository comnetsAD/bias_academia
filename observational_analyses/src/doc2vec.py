import gensim
import pandas as pd
from tqdm.notebook import tqdm

import collections
import glob
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def printShape(df, cols=[], msg=''):
    print(df.shape, end='  ')
    for col in cols:
        print(col, df[col].nunique(), end='  ')
    print(msg, flush=True)
    
    return df

# https://radimrehurek.com/gensim/models/callbacks.html#gensim.models.callbacks.CallbackAny2Vec
from gensim.models.callbacks import CallbackAny2Vec

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1

        if self.epoch % 10 == 1:
            output_path = f"../openalex/derived_data/abstract_doc2vec/EPOCH_{self.epoch}.model"
            model.save(output_path)


epoch_saver = EpochSaver()

# https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html

def read_corpus(df):
    
    for row in df:
        ID = row['PaperID']
        text = row['Abstract']
        
        tokens = gensim.utils.simple_preprocess(text)
        
        yield gensim.models.doc2vec.TaggedDocument(tokens, [ID])

def abstracts(files):

    for ind, file in enumerate(files):
        logging.info(f'Loading {ind}-th (out of {len(files)}) files ...')
        
        df = pd.read_csv(file).dropna().sample(frac=0.3, random_state=0).assign(PaperID=lambda df: df.PaperID.astype(str))
        
        for ind, row in df.iterrows():
            yield row

# abstracts = (
#     pd.concat(
#         [pd.read_csv(file, nrows=1000).dropna() for file in glob.glob('../openalex/cleaned_data/PaperAbstract/*.csv')],
#         ignore_index=True, sort=False
#     )
#     .pipe(printShape)
#     .assign(PaperID=lambda df: df.PaperID.astype(str))
# )

corpus = list(read_corpus(abstracts(glob.glob('../openalex/cleaned_data/PaperAbstract/*.csv'))))

print(f'{len(corpus)} abstracts in total', flush=True)

# docid2uid = list(abstracts.PaperID.unique())
# uid2docid = {docid2uid[ind]: ind for ind in range(len(docid2uid))}

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=100, epochs=40, workers=8)

model.build_vocab(corpus)

model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[epoch_saver])

model.save("../openalex/derived_data/abstract_doc2vec.model")



#### verification: find the closest description
# ranks = []
# second_ranks = []

# for doc_id in tqdm(range(len(corpus))):
    
#     inferred_vector = model.infer_vector(corpus[doc_id].words)
#     sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    
#     rank = [docid for docid, sim in sims].index(docid2uid[doc_id])
#     ranks.append(rank)

#     second_ranks.append(sims[1])

# counter = collections.Counter(ranks)
# print(counter)