from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, GenerationConfig, GPTNeoConfig, set_seed
from transformers import Trainer, TrainingArguments
from ChatData import ChatData
from torch.optim import Adam, Adadelta, Adamax, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import random
import wandb


FILENAME = "./Dataset/sheldon_chats.json"
random_seed_number = random.randint(0, 100)

config = {
    "learning_rate": 4e-5,
    "batch_size": 8,
    "project_name": "iNLP_Project",
    "entity_name": "mhardik003",
    "random_seed": random_seed_number,
}


# get a random number generator between 0 and 100
print("Seed  : ", random_seed_number)
set_seed(random_seed_number)


def init_wandb(selected_model, config):
    wandb.init(project=config["project_name"], entity=config["entity_name"], config=config,
               name=selected_model + "_" + str(config["random_seed"]))


def clean_output(text):
    """
    Clean the output text
    """
    text = text.replace("<START>", "Question :")
    text = text.replace("<bot>:", "\nSheldon :")
    text = text.replace("<END>", "")
    text = text.replace("<pad> ", "")
    text = text.replace("<pad>", "")

    return text


def train(chatData, model, optim, NUM_EPOCHS=10):
    """
    Train the model
    """
    for i in tqdm(range(NUM_EPOCHS)):
        model.train()

        if (i > 0):
            temp = infer("Hello, how are you?")
            with open("output.txt", 'a') as f:
                blabla = "Epoch " + \
                    str(i) + "\n" + str(temp) + \
                    "\n--------------------------------------------\n\n"
                # print(blabla)
                f.write(blabla)

        # change learning rate
        if (i % 3 == 0 and optim.param_groups[0]['lr'] > 1e-5):
            optim.param_groups[0]['lr'] /= 2

        for X, a in tqdm(chatData):
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()

            # calculate the loss of the model
            loss = model(X, attention_mask=a, labels=X).loss
            
           
            loss.backward()
            optim.step()

        ans_ques1 = infer("Hello, how are you?", 1)
        ans_ques2 = infer("What is your name?", 1)
        ans_ques3 = infer("Is your name Sheldon? Yes or No?", 1)

        torch.save(model.state_dict(), "model_state.pt")
        
        wandb.log({"loss": loss, "epoch": i, "learning_rate": optim.param_groups[0]['lr'], "Hello, how are you?": ans_ques1,
                   "What is your name?": ans_ques2, "Is your name Sheldon? Yes or No?": ans_ques3})

        print("-"*100)
        print("Hello, how are you? : ", ans_ques1)
        print("What is your name? : ", ans_ques2)
        print("Is your name Sheldon? Yes or No? : ",ans_ques3)

        with open("output.txt", 'a') as f:

            f.write("\n" + "-"*100 + "\n")
            f.write(ans_ques1 + "\n")
            f.write(ans_ques2 + "\n")
            f.write(ans_ques3 + "\n")


def infer(inp, f=0):
    """
    Infer from the model
    """
    # model.eval()
    inp = "<START> "+inp+"<bot>: "
    inp = tokenizer(inp, return_tensors="pt")

    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)

    output = model.generate(X, attention_mask=a)
    output = tokenizer.decode(output[0])

    if (f):
        index = output.find("<bot>:")
        output = output[index:]
    # model.train()
    return clean_output(output)


# check if cuda is available
device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device : ", device)

NUM_EPOCHS = input("Enter number of epochs : ")
model_type = input("Enter 1 for GPT2 and 2 for GPTNeo : ")


selected_model = "GPT2Medium" if model_type == "1" else "GPTNeo"
if model_type == "1":
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium", pad_token="<pad>", bos_token="<START>", eos_token="<END>")
    # config = model.config

else:
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M", pad_token="<pad>", bos_token="<START>", eos_token="<END>")


init_wandb(selected_model,config)

# add special tokens
tokenizer.add_tokens(["<bot>:"])

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 80
model.config.use_cache = True  # for faster generation of text
# model.config.repetition_penalty = 0.75 # how much to penalize for repeating words
# model.config.temperature = 0.85 # creativity setting (1 means most creative/random)
model.config.max_new_tokens = 100
model.config.attention_layers = ["global"] * 12

model = model.to(device)

chatData = ChatData(FILENAME, tokenizer)
chatData = DataLoader(chatData, batch_size=config["batch_size"], shuffle=True)

model.train()

optim = AdamW(model.parameters(), lr=config["learning_rate"])

print("training .... ")
train(chatData, model, optim, int(NUM_EPOCHS))

print("infer from model : ")
while True:
    inp = input("You :")
    print(infer(inp, 1))
