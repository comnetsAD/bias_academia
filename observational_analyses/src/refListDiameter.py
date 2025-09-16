import pandas as pd
import numpy as np
import glob
from tqdm.notebook import tqdm

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def printShape(df, cols=[], msg=''):

    message = f'{df.shape}'
    
    for col in cols:
        message += f'  {col} {df[col].nunique()}'

    message += '  '
    message += msg
    
    logging.info(message)
    
    return df

DIR = '/scratch/fl1092/Email_project/data'
OPENALEX = '/scratch/fl1092/OpenAlex/openalex'

PaperIDtoInt = lambda x: int(x.replace('https://openalex.org/W', ''))

totalRef = (
    pd.read_csv(f'{OPENALEX}/cleaned_data/PaperReferencesCount.csv')
    .pipe(printShape, msg='ref count loaded')

    .query('ReferencesCount >= 10')
    .pipe(printShape, msg='keeping papers with at least 10 refs')
    .drop('ReferencesCount', axis=1)
)

refs = (
    pd.read_csv(f'{OPENALEX}/cleaned_data/PaperReferences.csv')
    .pipe(printShape, msg='paper refs loaded')

    .merge(totalRef, on='PaperID')
    .pipe(printShape, msg='keeping papers with at least 10 refs')
    .rename(columns={'PaperID':'CitingPaper'})
)

def loadEmbedding(IND):
    
    embFile = f'{OPENALEX}/derived_data/abstract_doc2vec_embeddings/{IND}_embeddings.npy'
    IDFile = f'{OPENALEX}/derived_data/abstract_doc2vec_embeddings/{IND}_PaperIDs.csv'

    with open(embFile, 'rb') as f:
        embeddings = np.load(f, allow_pickle=True)

    IDs = pd.read_csv(IDFile, dtype={'PaperID':str})

    assert(len(embeddings) == IDs.shape[0])

    return embeddings, IDs


embeddingMap = {}

allPapers = []

for IND in range(258):
    embeddings, IDs = loadEmbedding(IND)

    for ind in range(len(embeddings)):
        embeddingMap[IDs.loc[ind,'PaperID']] = embeddings[ind]

    allPapers.append(IDs.pipe(printShape, msg=f'Loaded {IND}'))

allPapers = (
    pd.concat(allPapers, ignore_index=True, sort=False)
    .assign(PaperID=lambda df: df.PaperID.astype(int))
)

logging.info(f'{len(embeddingMap)} total papers {allPapers.shape}')

def calcDiameter(df):

    citingPaperID = str(df.CitingPaper.values[0])
    citingPaperVec = embeddingMap[citingPaperID]
    
    df = (
        df.reset_index()
        .assign(Vector=lambda df: df.BeingCited.apply(lambda x: embeddingMap[str(x)] if str(x) in embeddingMap else np.nan))
        .dropna()
    )
    centroid = df.Vector.mean()

    df = (
        df.assign(Diameter=lambda df: df.Vector.apply(lambda x: np.linalg.norm(x-centroid)))
        .assign(Distance=lambda df: df.Vector.apply(lambda x: np.linalg.norm(x-citingPaperVec)))
        .drop(['Vector','index'], axis=1)
    )

    return df

for ind, paperDf in enumerate(np.array_split(allPapers, 100)):

    distance = (
        refs.merge(paperDf.rename(columns={'PaperID':'CitingPaper'}), on='CitingPaper')
        .pipe(printShape, cols=['CitingPaper'], msg=f'{ind}-th slice')

        .groupby('CitingPaper')
        .apply(calcDiameter)
        .pipe(printShape, msg='slice done')
    )

    distance.to_csv(f'{DIR}/CitationsPaperDistance_30percent_allpapers/{ind}.csv', index=False)