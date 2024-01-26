from matplotlib import pyplot as plt
import model
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict
from tqdm import tqdm


def train(train_loss, log_interval):
    m.train()
    for batch_idx, batch in enumerate(train_loader):

        x, x_num = batch['x'].to(device), batch['x_num'].to(device)


        logit_pred, num_pred = m(x, x_num)

        #cross entropy loss between logits (tensor of lenght vocab_size) and the class (token) 
        l_mlm = loss_mlm(logit_pred.view(-1, logit_pred.size(-1)).to(device),
                    batch["y"].view(-1).to(device))
        
        num_mask = batch['y']==num_token_id
        
        #MSE loss between predicted numbers and actual numbers
        l_num = loss_num(num_pred[num_mask].to(device),
                    batch["y_num"][num_mask].view(-1,1).to(device))

        loss = l_mlm+l_num
        opt.zero_grad()

        loss.backward()
        opt.step()

        #to plot learning
        train_loss.append(loss.item())
        
        #print progress
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {e} [{batch_idx * len(x)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t Training Loss: {loss.data.item():.4f}')


def validate(val_loss):
    m.eval()

    average_loss = 0

    # weights not updated during validation
    with torch.no_grad():
        for batch in val_loader:

            x, x_num = batch['x'].to(device), batch['x_num'].to(device)


            logit_pred, num_pred = m(x, x_num)

            #cross entropy loss between logits (tensor of lenght vocab_size) and the class (token) 
            l_mlm = loss_mlm(logit_pred.view(-1, logit_pred.size(-1)).to(device),
                        batch["y"].view(-1).to(device))
            
            num_mask = batch['y']==num_token_id
            
            #MSE loss between predicted numbers and actual numbers
            l_num = loss_num(num_pred[num_mask].to(device),
                        batch["y_num"][num_mask].view(-1,1).to(device))

            loss = l_mlm+l_num

            average_loss += loss.item()


    #keep track of validation loss
    average_loss /= len(val_loader)
    print(f'\nValidation set average loss: {average_loss:.4f})\n\n')
    val_loss.append(average_loss)

def collator(batch):
    #ids to tensor
    x = [torch.tensor(sample["input_ids"]) for sample in batch]
    #all tensors to equal length, padded with pad_token_id
    x = pad_sequence(x, batch_first=True, padding_value=pad_token_id)

    #numbers to tensor
    x_num = [torch.tensor(sample["numbers"]) for sample in batch]
    #all tensors to equal length, padded with one (as described in paper)
    x_num = pad_sequence(x_num, batch_first=True, padding_value=1.)

    #create targets
    y = x.clone()
    y_num = x_num.clone().to(torch.float32)

    #mask tensors with mask_prob probability to create training data 
    mask = torch.rand(x.shape) < mask_prob
    x[mask] = mask_token_id
    x_num[mask] = 1.
    
    return {'x': x, 'x_num': x_num, 'y':y, 'y_num':y_num}



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using: {device}')
torch.manual_seed(42)

tok_path = "token/tokenizer.json"
data_path = "token/tokenized_ds"

learning_rate = 1e-4
epochs = 1
batch_size = 32
mask_prob = 0.3

#load tokenizer as done in tokenizer.py
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tok_path,
    bos_token="[END]",
    eos_token="[END]",
    mask_token="[MASK]",
    pad_token="[PAD]"
)
pad_token_id = tokenizer.pad_token_id
mask_token_id = tokenizer.mask_token_id
num_token_id = tokenizer.convert_tokens_to_ids('[NUM]')

#load dataset
tokenized_ds = DatasetDict.load_from_disk(data_path)

#train and val loaders, collator is applied to the data
train_loader = DataLoader(
    tokenized_ds['train.txt'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collator
)
val_loader = DataLoader(
    tokenized_ds['val.txt'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collator
)
    
#model, losses and optimizer
vocab_size = len(tokenizer.vocab)
m = model.Numformer(vocab_size=vocab_size).to(device)
opt = optim.Adam(m.parameters(), lr=learning_rate)
loss_mlm = nn.CrossEntropyLoss()
loss_num = nn.MSELoss()

#training and validation
log_interval = 32
train_loss = []
val_loss = []
for e in tqdm(range(epochs)):
    train(train_loss, log_interval)
    validate(val_loss)

# matplotlib plot
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
