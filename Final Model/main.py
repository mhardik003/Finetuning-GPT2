from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM
from ChatData import ChatData
from torch.optim import Adam, Adadelta
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


def train(chatData, model, optim):
    epochs = 10
    for i in tqdm(range(epochs)):
        # change learning rate
        if (i % 4 == 0):
            optim.param_groups[0]['lr'] /= 2
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
    output = model.generate(X, attention_mask=a)
    output = tokenizer.decode(output[0])
    output = output.replace("<START>", "Question :")
    output = output.replace("<bot>:", "\nSheldon :")
    output = output.replace("<END>", "")
    output = output.replace("<pad> ", "")
    return output


device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.add_special_tokens({"pad_token": "<pad>",
                              "bos_token": "<START>",
                              "eos_token": "<END>"})
tokenizer.add_tokens(["<Sheldon>:"])
tokenizer.pad_token = "<pad>"
tokenizer.bos_token = "<START>"
tokenizer.eos_token = "<END>"

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 300
model.config.max_new_tokens = 300


print("Using device : ", device)
model = model.to(device)

# print(tokenizer.decode(model.generate(**tokenizer("hey i was good at basketball but ",
#                          return_tensors="pt"))[0]))

chatData = ChatData("./Dataset/sheldon_chats_bigger_context.json", tokenizer)
chatData = DataLoader(chatData, batch_size=2)

model.train()

optim = Adam(model.parameters(), lr=1e-2)

print("training .... ")
train(chatData, model, optim)

print("infer from model : ")
while True:
    inp = input("You :")
    print(infer(inp))
