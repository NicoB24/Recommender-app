import pandas as pd
import pickle
from scipy import spatial

class ItemsPredictor:
    def __init__(self):
        ## Metadata loading
        prep_files = 'meli_files/prep_files.pkl'

        with open(prep_files, "rb") as f:
            df_items_selec, le_x, embeddings = pickle.load(f)

        self.df_items_selec = df_items_selec
        self.le_x = le_x
        self.embeddings = embeddings
    
        ## Tree parameters
        self.tree = spatial.KDTree(embeddings)
        self.k_elements = 3

    def predict(self, data):
        encoded_id_to_check = int(data['item_1'])

        df_most_similar = self.__find_most_similar_items(encoded_id_to_check)
    
        # Target item
        df_target_item = self.__target_item(self.df_items_selec, encoded_id_to_check)

        # Related items
        df_items_rel = self.__related_items(self.df_items_selec, df_most_similar)
    
        return self.__result_dataset(df_target_item, df_items_rel)

    def __find_most_similar_items(self, encoded_id_to_check):
        dist, encoded_id = self.tree.query(self.embeddings[encoded_id_to_check], k=self.k_elements+1)

        df_most_similar = pd.DataFrame({'dist':dist, 'encoded':encoded_id})

        df_most_similar = df_most_similar[df_most_similar['encoded'] != encoded_id_to_check].copy()

        df_most_similar['rank'] = list(range(1, self.k_elements+1))

        similar_item_id = self.le_x.inverse_transform(df_most_similar['encoded'])
        df_most_similar['item_id'] = similar_item_id

        return df_most_similar

    def __target_item(self, df_items_selec, encoded_id_to_check):
        return  df_items_selec[df_items_selec['item_id'].isin(self.le_x.inverse_transform([encoded_id_to_check]))][['item_id','title', 'domain_id', 'price']]

    def __related_items(self, df_items_selec, df_most_similar):
        df_top_items = df_items_selec[df_items_selec['item_id'].isin(df_most_similar['item_id'])][['item_id','title', 'domain_id', 'price']]

        return pd.merge(df_top_items, df_most_similar, on='item_id').sort_values('rank')

    def __result_dataset(self, df_target_item, df_items_rel):
        result = {
        'item_base': 'Item ID: ' + str(df_target_item.iloc[0]['item_id']) + ' | ' + str(df_target_item.iloc[0]['title']), 
    
        'item_reco_1': 'Item ID: ' + str(df_items_rel['item_id'][0]) + ' | Encoded: ' + str(df_items_rel['encoded'][0]) + ' | ' + str(df_items_rel['title'][0]), 
        'item_reco_2': 'Item ID: ' + str(df_items_rel['item_id'][1]) + ' | Encoded: ' + str(df_items_rel['encoded'][1]) + ' | ' + str(df_items_rel['title'][1]), 
        'item_reco_3': 'Item ID: ' + str(df_items_rel['item_id'][2]) + ' | Encoded: ' + str(df_items_rel['encoded'][2]) + ' | ' + str(df_items_rel['title'][2])
        }

        return result
