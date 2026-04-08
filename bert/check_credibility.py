import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# 1. Load Dataset
# Example CSV should have columns: 'text' and 'label' (0 = real, 1 = fake)
df = pd.read_csv("fake_news.csv")

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 2. Load Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 3. Load Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 6. Train Model
trainer.train()

# 7. Evaluate
trainer.evaluate()

# 8. Prediction Function
def predict(text):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model.to(device)  # move model to device
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # move inputs to same device
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits).item()
    
    return "Fake News" if predicted_class_id == 1 else "Real News"

# Example usage
print(predict("Breaking: Scientists discovered water on Mars"))
print(predict("Breaking: Scientists discovered water on Sun"))


# import transformers
# print("Transformers version:", transformers.__version__)
# print("Path:", transformers.__file__)