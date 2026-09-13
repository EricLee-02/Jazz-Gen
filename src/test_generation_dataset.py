# from jazz_generation_dataset import JazzGenerationDataset


# dataset=JazzGenerationDataset(

#     solo_dir=
#     "/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2_token/token/",

#     vocab_file=
#     "/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2_token/vocabulary/vocabulary.json"

# )


# print(len(dataset))


# sample=dataset[0]


# print(sample["harmony"]["input_ids"].shape)

# print(sample["melody_input"].shape)

# print(sample["melody_target"].shape)

from torch.utils.data import DataLoader

from jazz_generation_dataset import JazzGenerationDataset



dataset=JazzGenerationDataset(solo_dir="/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2_token/token/",
    vocab_file= "/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2_token/vocabulary/vocabulary.json"
)

loader=DataLoader(dataset,batch_size=8,shuffle=True)
batch=next(iter(loader))
print(batch.keys())
print(batch["harmony"]["input_ids"].shape)
print(batch["melody_input"].shape)
print(batch["melody_target"].shape)