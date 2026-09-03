import json
from pathlib import Path
import random 
import torch
from torch.utils.data import Dataset, DataLoader


class JazzDataset(Dataset):
    def __init__(self,json_dir,vocab_file,seq_length=512,stride=256,split="train",train_ratio=0.9,seed=42):
        self.seq_length = seq_length
        # vocab
        with open(vocab_file,"r") as f:
            vocab=json.load(f)
        self.token_to_id = vocab["token_to_id"]
        self.id_to_token = {int(k):v for k,v in vocab["id_to_token"].items()}
        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]
        self.data=[]
        files=list(Path(json_dir).glob("*.json"))
        print("Total Songs:",len(files))

        random.seed(seed)
        random.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        if split == "train":
            files = files[:split_idx]
        else:
            files = files[split_idx:]
        print(split, "songs:", len(files))


        for file in files:
            try :
                with open(file,"r",encoding="utf-8") as f:song=json.load(f)
                tokens=song["tokens"]
                ids=[]
                for token in tokens:
                    ids.append(self.token_to_id.get(token,self.unk_id))
                ids = [self.bos_id] + ids + [self.eos_id]
                for start in range(0,len(ids)-seq_length,stride):
                    chunk = ids[start:start+seq_length]
                    self.data.append(chunk)
                if len(ids) < seq_length :
                    self.data.append(ids)
                # self.data.append(ids)
            except Exception as e:
                print("Bad Json", file, e)
        print("Loaded sequences:",len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        ids=self.data[index]
        original_length = len(ids)
        # 保证长度
        if len(ids)>self.seq_length:
            ids=ids[:self.seq_length]
        else:
            ids=ids + [self.pad_id] * (self.seq_length-len(ids))

        input_ids=torch.tensor(ids[:-1],dtype=torch.long)
        labels=torch.tensor(ids[1:],dtype=torch.long)

        attention_mask =torch.tensor( [0 if token == self.pad_id else 1 for token in ids[:-1]],dtype= torch.long)
        length = sum(1 for token in ids[:-1] if token != self.pad_id)
        return {
            "input_ids":input_ids,
            "labels":labels,
            "length": original_length,
            "attention_mask": attention_mask
        }
    

if __name__=="__main__":
    json_dir = '/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2_token/token'
    vocab_file = '/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2_token/vocabulary/vocabulary.json'


    train_dataset=JazzDataset(json_dir,vocab_file,split="train")
    train_loader=DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=0)
    val_dataset = JazzDataset(json_dir,vocab_file,split="val")
    batch=next(iter(train_loader))
    print("input_ids:")
    print(batch["input_ids"].shape)
    print("labels:")
    print(batch["labels"].shape)
    print("attention_mask:")
    print(batch["attention_mask"].shape)

    # print("\n========== First Sample ==========")
    # sample=train_dataset[0]
    # print("\nTrain sample")
    # print("input:",sample["input_ids"].shape)
    # print("labels:",sample["labels"].shape)
    # print("mask:",sample["attention_mask"].shape)
    # print(sample["attention_mask"][:20])



    # input_ids=sample["input_ids"]
    # labels=sample["labels"]
    # print("Original length:",sample["length"])
    # print("Input shape:",input_ids.shape)
    # print("Label shape:",labels.shape)
    # print("\nFirst 50 input ids:")
    # print(input_ids[:50].tolist())
    # print("\nDecode tokens:")
    # for idx in input_ids[:50]:
    #     token=dataset.id_to_token[(idx.item())]
    #     print(token)
    # print("\nFirst 50 labels:")
    # for idx in labels[:50]:
    #     print(dataset.id_to_token[(idx.item())])