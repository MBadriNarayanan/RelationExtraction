import re

from torch.utils.data import Dataset


class RelationDataset(Dataset):
    def __init__(
        self,
        dataframe,
    ):
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]["context"].strip()
        text = re.sub("([\[\].,!?()])", r" \1 ", text.replace("()", ""))
        text = re.sub("\s{2,}", " ", text)

        relation = self.dataframe.iloc[idx]["triplets"].strip()
        return (text, relation)
