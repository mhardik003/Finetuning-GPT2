from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ChatData import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch

def train(chatData, model, optim):

    epochs = 20

    for i in tqdm.tqdm(range(epochs)):
        for X, a in chatData:
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
        torch.save(model.state_dict(), "model_state.pt")
        print(infer("Hello, how are you?"))

def infer(inp):
    inp = "<START> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a )
    output = tokenizer.decode(output[0])
    output = output.replace("<START>", "Question :")
    output = output.replace("<bot>:", "\nSheldon :")
    output = output.replace("<END>", "")
    output = output.replace("<pad> ", "")
    return output


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<START>",
                                "eos_token": "<END>"})
tokenizer.add_tokens(["<Sheldon>:"])
tokenizer.pad_token = "<pad>"
tokenizer.bos_token = "<START>"
tokenizer.eos_token = "<END>"

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 100
model.config.max_new_tokens = 100

model = model.to(device)

# print(tokenizer.decode(model.generate(**tokenizer("hey i was good at basketball but ",
#                          return_tensors="pt"))[0]))

chatData = ChatData("./Dataset/sheldon_chats.json", tokenizer)
chatData =  DataLoader(chatData, batch_size=64)

model.train()

optim = Adam(model.parameters(), lr=1e-3)

print("training .... ")
train(chatData, model, optim)

print("infer from model : ")
while True:
  inp = input("You :")
  print(infer(inp))