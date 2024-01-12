import model
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using: {device}')

tok_path = "token/tokenizer.json"
data_path = "token/tokenized_ds"

learning_rate = 1e-4
epochs = 10
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

def collator(batch):
    x = [torch.tensor(sample["input_ids"]) for sample in batch]
    x = pad_sequence(x, batch_first=True, padding_value=pad_token_id)

    x_num = [torch.tensor(sample["numbers"]) for sample in batch]
    x_num = pad_sequence(x_num, batch_first=True, padding_value=1.)

    y = x.clone()
    y_num = y.clone().to(torch.float32)

    mask = torch.rand(x.shape) < mask_prob
    x[mask] = mask_token_id
    x_num[mask] = 1.
    
    return {'x': x, 'x_num': x_num, 'y':y, 'y_num':y_num}


tokenized_ds = DatasetDict.load_from_disk(data_path)
train_loader = DataLoader(
    tokenized_ds['train.txt'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collator
)


vocab_size = len(tokenizer.vocab)
m = model.Numformer(vocab_size=vocab_size).to(device)
opt = optim.Adam(m.parameters(), lr=learning_rate)
loss_mlm = nn.CrossEntropyLoss()
loss_num = nn.MSELoss()

log_interval = 32
tot_loss = []
mlm_loss = []
num_loss = []
m.train()
for e in tqdm(range(epochs)):
    for batch_idx, batch in enumerate(train_loader):

        x, x_num = batch['x'].to(device), batch['x_num'].to(device)


        logit_pred, num_pred = m(x, x_num)


        l_mlm = loss_mlm(logit_pred.view(-1, logit_pred.size(-1)).to(device),
                    batch["y"].view(-1).to(device))
        
        num_mask = batch['y']==num_token_id
        l_num = loss_num( num_pred[num_mask].to(device),
                    batch["y_num"][num_mask].view(-1,1).to(device))

        loss = l_mlm+l_num
        opt.zero_grad()

        loss.backward()
        opt.step()

        tot_loss.append(loss.item())
        mlm_loss.append(l_mlm.item())
        num_loss.append(l_num.item())

        
        #ogni tanto stampa loss di questo batch
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {e} [{batch_idx * len(x)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data.item():.4f}')
