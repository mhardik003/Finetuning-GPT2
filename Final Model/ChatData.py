from torch.utils.data import Dataset
import json

class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = json.load(open(path, "r"))

        self.X = []
        for i in self.data:
            for j in i['dialog']: 
                self.X.append(j['text'])

        for idx, i in enumerate(self.X):
            try:
                if((len(self.X[idx].split(' ')) + len(self.X[idx+1].split(' '))) > 250):
                        
                        to_be_truncated = len(self.X[idx].split(' ')) + len(self.X[idx+1].split(' ')) - 250
                        # print(to_be_truncated)
                        if(idx>4 and idx<10):
                            print(self.X[idx])
                            print("-"*20)
                            print(" ".join(self.X[idx].split(' ')[:-to_be_truncated]))
                        self.X[idx] = " ".join(self.X[idx].split(' ')[:-to_be_truncated]) 
                        i = self.X[idx] + " \n "
                        
                self.X[idx] = "<START> "+i+" <bot>: "+self.X[idx+1]+" <END>"
                
                if(idx>=9 and idx<21):
                    print("----------------------------------------------------------------------------------------")
                    print(self.X[idx])
                    
                idx+=1
                        
            except:
                break
        
        # print(len(self.X))
        self.X = self.X[:2000]
        
        # print(self.X[0])

        self.X_encoded = tokenizer(self.X,max_length=300, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])