from datasets import load_dataset

dataset = load_dataset("FronkonGames/steam-games-dataset")

# get columns names and types
columns = dataset["train"].features
print(columns)

columns_to_keep = ["name", "windows", "linux", "mac", "detailed_description", "supported_languages", "price"]

N = 40000 # you can adjust this number
dataset = dataset["train"].select_columns(columns_to_keep).select(range(N))
