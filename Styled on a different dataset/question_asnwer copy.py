# load the pt file
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ChatData import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<pad>",
                              "bos_token": "<START>",
                              "eos_token": "<END>"})
tokenizer.add_tokens(["<bot>:"])
tokenizer.pad_token = "<pad>"
tokenizer.bos_token = "<START>"
tokenizer.eos_token = "<END>"


device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 100
model.config.max_new_tokens = 100


model.load_state_dict(torch.load('model_state.pt'))
model = model.to(device)
model.eval()


def infer(inp):
    inp = "<START> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a)
    output = tokenizer.decode(output[0])
    index = output.find("<bot>:")
    output = output.replace("<START>", "Question :")
    output = output[index:]
    output = output.replace("<bot>:", "\nSheldon :")
    output = output.replace("<END>", "")
    # output = output.replace("<pad>", "")
    return output



while True:
    inp = input("You: ")
    if inp == "exit":
        break
    print(infer(inp))
    
    
