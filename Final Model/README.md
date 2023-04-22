Transformer (GPT-2) fine tuned on the chat_data.json dataset

Change log

* (20/03/2023) :
    - Made some basic changes and played around with the model using hyperparameter tuning

<br>

* (21/03/2023) :
    - Changed the model to a GPT2-Medium model
    - Changed the dataset to cover more context
    - Did some hyperparameter tuning according to the new dataset
    - File structure changes ( all the new changes will be done in Final Model folder and not in the 'Styled on a different dataset' folder)


* (21/04/2023) : 
  - Added a seed to see which one performs the best
  - Added WandB to store the models and compare their performances ( https://wandb.ai/mhardik003/iNLP_Project)
  - Changed the optimizer to AdamW as recommended by a lot of online resources
  - Changed the learning rate to 1e-5 as recommended by a lot of online resources
  

* (22/04/2023) :
 - Added scheduler
 - Froze the first 6 layers of the model