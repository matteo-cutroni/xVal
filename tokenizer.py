from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models
import os
import re

#gets keys of a dictionary
def extract_keys(string):
    pattern = r"'(.*?)'"
    return re.findall(pattern, string)



#returns dictionary with h_text and h_num embeddings of the sample
def tokenize_fnc(sample, tokenizer, num_token="[NUM]"):
    num_tok_id = tokenizer.convert_tokens_to_ids('[NUM]')

    #when doing map() sample is a dictionary
    if type(sample) != str:
        sample = sample["text"]
    
    # this regular expression is intended to match numerical values in various forms
    # like integers, floating-point numbers, or scientific notation, while avoiding
    # matching numbers that are part of strings.
    pattern = r"(?<!\')-?\d+(\.\d+)?([eE][-+]?\d+)?(?!\'|\d)"
    
    #appends in numbers what is replaced by the num_token according to the re above
    numbers = []
    def replace(match):
        numbers.append(float(match.group()))
        return num_token
    #substitutes each number in the sample with the num_token
    nonum_text = re.sub(pattern, replace, sample)
    #returns a dictionary (False options to not get unnecessary keys:value pairs)
    out = tokenizer(nonum_text, return_attention_mask=False, return_token_type_ids=False)

    #h_text in the paper (array of token ids of the sample) 
    ids = out['input_ids']
    #h_num in the paper (similar to ids but has the value of the number instead of the num_tok_id. Other tokens are replaced with ones)
    num_embed = [1.]*len(ids)
    j=0
    for i in range(len(ids)):
        if ids[i] == num_tok_id:
            num_embed[i] = numbers[j]
            j+=1

    #add number embedding to the dictionary
    out['numbers'] = num_embed
    out['len'] = len(ids)
    return out
    

            

#paths
data_path = 'data/'
save_path = 'token/'
tok_path = save_path + 'tokenizer.json'
tok_ds_path = save_path +'tokenized_ds'


files = os.listdir(data_path)
#dictionary to store the dataset (has as keys the file names and as values the datasets)
ds = DatasetDict.from_text({file: data_path+"/"+ file for file in files})

#ds['train.txt']['text'] contains the actual data
#get the keys for each sample in train data ('text' column is not needed)
ds_keys = ds['train.txt'].map(
    lambda x: {"keys": extract_keys(x['text'])},
    remove_columns=['text']
)

#list of all keys in train data without duplicates (reason for list(set()))
sample_keys =  list(set(item for sublist in ds_keys['keys'] for item in sublist ))

#special tokens to be added in the tokenizer
special_tokens = ["[END]", "[MASK]", "[PAD]", "[NUM]"]
#non special tokens to be added (keys extracted before)
vocab=[]
for key in sample_keys:
    vocab+=[f"'{key}':"]
tokenizer = Tokenizer(models.BPE())
tokenizer.add_special_tokens(special_tokens)
tokenizer.add_tokens(vocab)
#save tokenizer config to be loaded from PreTrainedTokenizerFast
tokenizer.save(tok_path)

#PreTrainedTokenizerFast so that it can be later called
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tok_path,
    bos_token="[END]",
    eos_token="[END]",
    mask_token="[MASK]",
    pad_token="[PAD]"
)

#tokenize the datasets
tokenized_ds = ds.map(
    lambda x: tokenize_fnc(x, tokenizer),
    remove_columns=["text"],
)
#save tokenized datasets
tokenized_ds.save_to_disk(tok_ds_path)