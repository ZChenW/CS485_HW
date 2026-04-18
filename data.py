from datasets import load_dataset

ds = load_dataset("Pclanglais/Brahe-Novels")
ds.save_to_disk("data/brahe_novels")
