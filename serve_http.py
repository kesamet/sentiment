"""
Script for serving.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# from flask import Flask, request

MAX_LEN = 256
FINETUNED_MODEL_PATH = 'models/pytorch_distilbert_news2.bin'

TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

DEVICE = torch.device("cpu")
MODEL = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-cased', num_labels=2)
MODEL.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE))
MODEL.eval()


class TextDataset(Dataset):
    """Text dataset."""
    def __init__(self, data, tokenizer, max_len):
        self.len = len(data)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        sentence = self.data[index]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }

    def __len__(self):
        return self.len


# pylint: disable=too-many-locals
def predict(request_json):
    """Predict function."""
    sentences = request_json["sentences"]
    test_dataset = TextDataset(sentences, TOKENIZER, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    y_prob = list()
    with torch.no_grad():
        for data in test_loader:
            ids = data['ids'].to(DEVICE)
            mask = data['mask'].to(DEVICE)

            logits = MODEL(ids, attention_mask=mask)[0]
            probs = torch.softmax(logits, axis=1)
            y_prob.extend(probs[:, 1].cpu().numpy().tolist())
    return y_prob


# # pylint: disable=invalid-name
# app = Flask(__name__)
#
#
# @app.route("/", methods=["POST"])
# def get_prob():
#     """Returns probability."""
#     y_prob = predict(request.json)
#     return {"y_prob": y_prob}
#
#
# def main():
#     """Starts the Http server"""
#     # app.run()
#
#
# if __name__ == "__main__":
#     main()
