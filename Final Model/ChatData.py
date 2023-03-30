from torch.utils.data import Dataset
import json

class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):

        self.data = json.load(open(path, "r"))

        self.X = []
        for i in self.data:
            for j in i['dialog']: 
                self.X.append(j['text'])


        X_new=[]
        
        for idx in range(0,len(self.X),2):
            try:
                i = self.X[idx]
                if((len(self.X[idx].split(' ')) + len(self.X[idx+1].split(' '))) > 100):
                        
                        to_be_truncated = len(self.X[idx].split(' ')) + len(self.X[idx+1].split(' ')) - 100
                        # print(to_be_truncated)

                        # if(idx>=10 and idx<20):
                        #     print(self.X[idx])
                        #     print("-"*20)
                        #     print(" ".join(self.X[idx].split(' ')[:-to_be_truncated]))
                        self.X[idx] = " ".join(self.X[idx].split(' ')[:-to_be_truncated])
                        
                        # print(self.X[idx])
                        # print("-"*20)
                        i = self.X[idx]
                        
                string_feed= "<START> "+i+"\n<bot>: "+self.X[idx+1]+" <END>"
                
                if(len(string_feed.split(" "))>=300):
                    print(idx)
                    print(string_feed)
                    print("----------------------------------------------------------------------------------------\n\n")


                X_new.append(string_feed)
                
                # if(idx>=0 and idx<20):
                #     print("----------------------------------------------------------------------------------------")
                #     print(self.X[idx])
                #     print(X_new[-1])
                                            
            except:
                break
        
        # print(len(self.X))
        self.X=X_new

        for x in range(0,20):
            print(self.X[x])
            print("-"*20)

        self.X = self.X[:2000]
        maxi = 0 
        for x in self.X:
            maxi = max(maxi, len(tokenizer(x)['input_ids']))
        print("asdfasdfasd"+str(maxi)+"\n\n\n") 
        
        # print(self.X[0])

        self.X_encoded = tokenizer(self.X,max_length=300, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])