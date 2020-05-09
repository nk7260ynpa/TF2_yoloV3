import numpy as np
import pandas as pd

class imagenet_info():
    def __init__(self, map_index_path, val_label_path):
        self.map_index_path = map_index_path
        self.val_label_path = val_label_path
        self.df = self._df_generator()
        self.val_label = self._valid_label_generator()
        self.index_to_net_label = dict(zip(self.df["index"], self.df["net_label"]))
        self.label_to_net_label = dict(zip(self.val_label["label"], self.val_label["net_label"]))
        self.net_label_to_index = dict(zip(self.val_label["net_label"], self.val_label["index"]))
    
    def _df_generator(self):
        with open(self.map_index_path) as f:
            map_index = f.readlines()
        df = pd.DataFrame(map_index, columns=["all_label"])
        df["index"] = df.all_label.str.split(" ").str.get(0)
        df["label"] = df.all_label.str.split(" ").str.get(1).astype(np.int32)
        df["name"] = df.all_label.str.split(" ").str.get(2).str.strip("\n")
        df["net_label"] = df["label"] - 1
        df = df.drop("all_label", axis=1)
        df = df[["net_label", "label", "index", "name"]]
        return df
    
    def _valid_label_generator(self):
        with open(self.val_label_path) as f:
            val_label = f.readlines()
        val_label_df = pd.DataFrame(val_label, columns=["label"])
        val_label_df["label"] = val_label_df["label"].str.strip("\n").astype(np.int32)
        image_num = [i for i in range(1, 50001)]
        val_label_df["image_num"] = image_num
        val_label_df = val_label_df.merge(self.df, left_on='label', right_on='label')
        val_label_df = val_label_df.sort_values("image_num")
        val_label_df = val_label_df.reset_index(drop=True)
        val_label_df = val_label_df[["image_num", "net_label", "label", "index", "name"]]
        return val_label_df 
      