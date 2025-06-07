import pandas as pd

class Curriculum:
    def __init__(self, path: str, subjects: pd.DataFrame):
        self.df = pd.read_csv(path)

        self.df = self.df.merge(subjects, left_on='subject_id', right_on='id', how='left', suffixes=('_curr', '_subj'))
        self.df.drop(columns=['id_subj', 'subject_id'], inplace=True)
        self.df.rename(columns={
            'id_curr': 'id',
            'code': 'subject_code',
            'subject': 'subject_name'
        }, inplace=True)
    
    