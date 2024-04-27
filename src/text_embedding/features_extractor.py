from transformers import BertTokenizer, BertModel

from torch import cuda

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()

class FeaturesExtractor():
    '''
    Extracts from texts a vector of features
    '''
    def __init__(self):
        self.device = "cuda:0" if cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("WARNING: BERT is on cpu.")

        self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = BertModel.from_pretrained("prajjwal1/bert-tiny")
        self.model = self.model.to(self.device)

        self.tokenizer.model_max_length = 512
        self.tokenizer.truncation_side = 'right'

    def extract(self, text):
        encoded_input = self.tokenizer(
            text, return_tensors='pt', truncation=True).to(self.device)
        output = self.model(**encoded_input)

        return output.pooler_output.tolist()[0]
    
def extract_features_from_small_graphs(sg_path, only_title=False):
    df = pd.read_parquet(sg_path, engine='fastparquet')
    fe = FeaturesExtractor()

    def process(row):
        if only_title:
            text = row['title']
        else:
            text = row['abstract']
            if text is None or text == '':
                text = row['title']

        if text is None or text == '':
            print(f"WARNING: row with id: {row['id']} has invalid text.")

        features = fe.extract(text)
        y = 1 if row['has_award'] else 0

        return features, y, text == row['title']

    data = df.progress_apply(lambda x: process(x), axis=1, result_type="expand")
    X = np.array(data[0].values.tolist())
    y = data[1].to_numpy()

    from_title = len(data[data[2]])

    print(f"Extracted {from_title} features from titles and {len(y) - from_title} from abstracts.")

    return X, y