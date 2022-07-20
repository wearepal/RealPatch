import gdown
from pathlib import Path

output_dir = Path().resolve() / "data"
link_labels_train = "https://drive.google.com/uc?id=1A0bOhSqhGwdUUZ1LYmW9CzK_eis3rQvD"
link_labels_val = "https://drive.google.com/uc?id=1VNOl3Kzg68TNc0Zdd5rypUH7hZiSATmF"
link_labels_test = "https://drive.google.com/uc?id=1tjuKwDZ9QuLiCA2uKXhEIB9MxnHURUud"
link_features = "https://drive.google.com/uc?id=1fJdwniKFk_TnqwxmE_4POJgV6D59NsxK"

save_file_names = ["labels_train.csv", "labels_val.csv", "labels_test.csv", "features_file.npz"]
link_list = [link_labels_train, link_labels_val, link_labels_test, link_features]
for link, name in zip(link_list, save_file_names):
    gdown.download(link, str(output_dir / name))
