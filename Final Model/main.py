from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, GenerationConfig, GPTNeoConfig
from ChatData import ChatData
from torch.optim import Adam, Adadelta, Adamax
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch



FILENAME = "./Dataset/sheldon_chats_bigger_context.json"

def clean_output(text):
    text = text.replace("<START>", "Question :")
    text = text.replace("<bot>:", "\nSheldon :")
    text = text.replace("<END>", "")
    text = text.replace("<pad> ", "")
    return text


def train(chatData, model, optim, NUM_EPOCHS=10):
    
    for i in tqdm(range(NUM_EPOCHS)):
        # model.train()
        temp=infer("Hello, how are you?")

        with open("output.txt", 'a') as f:
            blabla = "Epoch " + str(i) + "\n" + str(temp) + "\n--------------------------------------------\n\n"
            print(blabla)
            f.write(blabla)
            
            
        # change learning rate
        # if (i % 3 == 0 and optim.param_groups[0]['lr'] > 1e-3):
            # optim.param_groups[0]['lr'] /= 2
        for X, a in tqdm(chatData):
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            
            if (i % 50 == 0):
                print('-'*100)
                print(tokenizer.decode(X[0]))
            
            
            # calculate the sentence after <bot>: token
            response = X[:, X[0].tolist().index(tokenizer("<bot>:").input_ids[0]):]
            
            # pad the response to the max length of the input
            response = torch.nn.functional.pad(response, (0, X.shape[1]-response.shape[1]), value=tokenizer.pad_token_id)
            
            print(response.shape, X.shape)
            # print decoded response
            # # print(tokenizer.decode(X[0]))
            # if (tokenizer.decode(X[0]).find("<bot>:") == -1):
            #     print   (tokenizer.decode(X[0]))
                
            
            
            
            # calculate the output of the model 
            output = model(X, attention_mask=a, labels=response)
            loss = output.loss
            
            # output = model(X, attention_mask=a, labels=X) 
            # compare only the sentence after <bot>: token with the actual output to calculate the loss

            
            loss.backward()
            optim.step()
            
        torch.save(model.state_dict(), "model_state.pt")
        print(infer("Hello, how are you?"))



def infer(inp):
    # model.eval()
    inp = "<START> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a)
    output = tokenizer.decode(output[0])
    # model.train()
    return clean_output(output)



device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device : ", device)

NUM_EPOCHS = input("Enter number of epochs : ")

model_type = input("Enter 1 for GPT2 and 2 for GPTNeo : ")
if model_type == "1":
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = model.config

else:
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    config =  GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    config.attention_layers = ["global"] * 12
    config.temperature = 0.5 # creativity setting (1 means most creative/random)
    config.use_cache = True
    config.repetition_penalty = 0.75 # how much to penalize for repeating words
    model.config = config


tokenizer.add_special_tokens({"pad_token": "<pad>",
                              "bos_token": "<START>",
                              "eos_token": "<END>"})
tokenizer.add_tokens(["<bot>:"])


model.resize_token_embeddings(len(tokenizer))




model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 20
model.config.use_cache = True # for faster generation of text
# model.config.repetition_penalty = 0.75 # how much to penalize for repeating words
# model.config.temperature = 0.35 # creativity setting (1 means most creative/random)
model.config.max_new_tokens = 300



# generation_config = GenerationConfig(
#     max_new_tokens=300,
#     max_length=20,
#     pad_token_id=tokenizer.pad_token_id,
#     bos_token_id=tokenizer.bos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     use_cache=True,
# )



model = model.to(device)

# print(tokenizer.decode(model.generate(**tokenizer("hey i was good at basketball but ",
#                          return_tensors="pt"))[0]))


chatData = ChatData(FILENAME, tokenizer)
chatData = DataLoader(chatData, batch_size=2)

model.train()

optim = Adam(model.parameters(), lr=1e-5)

print("training .... ")
train(chatData, model, optim, int(NUM_EPOCHS))

print("infer from model : ")
while True:
    inp = input("You :")
    print(infer(inp))
