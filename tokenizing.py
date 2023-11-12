import pandas as pd
from konlpy.tag import Kkma, Okt, Hannanum  

class Preprocessor:
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(data, encoding='utf-8')
        self.okt = Okt()
        self.kkma = Kkma()
        self.hannanum = Hannanum()
        




    def tokenizing(self, data):
        data['comments'] = data['comments'].astype(str)
        tok_okt_morphs = data['comments'].apply(lambda row: ' '.join(self.okt.morphs(row)))
    

        tok_okt_pos = data['comments'].apply(lambda row: ' '.join(['{}/{}'.format(word, pos) for word, pos in self.okt.pos(row)]))
    
        tok_okt_nouns = data['comments'].apply(lambda row: ' '.join(self.okt.nouns(row)))
        
    
        tok_kkma_morphs = data['comments'].apply(lambda row: ' '.join(self.kkma.morphs(row)))
        
    
        tok_kkma_pos = data['comments'].apply(lambda row: ' '.join(['{}/{}'.format(word, pos) for word, pos in self.kkma.pos(row)]))
        
    
        tok_kkma_nouns = data['comments'].apply(lambda row: ' '.join(self.kkma.nouns(row)))
        
    
        tok_hannaum_morphs = data['comments'].apply(lambda row: ' '.join(self.hannanum.morphs(row)))

        tok_hannaum_pos = data['comments'].apply(lambda row: ' '.join(['{}/{}'.format(word, pos) for word, pos in self.hannanum.pos(row)]))
            
        tok_hannaum_nouns = data['comments'].apply(lambda row: ' '.join(self.hannanum.morphs(row)))
        return tok_okt_morphs, tok_okt_pos, tok_okt_nouns, tok_kkma_morphs, tok_kkma_pos, tok_kkma_nouns, tok_hannaum_morphs, tok_hannaum_pos, tok_hannaum_nouns
    





