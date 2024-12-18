import os
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset

# Suppress parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# File paths
training_data_file = "/Users/vincentgruben/documents/Masterarbeit/bio_dataset6.csv"
job_descriptions_file = "/Users/vincentgruben/documents/Masterarbeit/test_file2.csv"
output_file = "/Users/vincentgruben/documents/Masterarbeit/NER_dataset_test.csv"

# Combined Custom and Standard BIO-Tags
CUSTOM_TAGS = [
    "B-ROLE", "I-ROLE", "B-COMPANY", "I-COMPANY", "B-LOC", "I-LOC",
    "B-ADDRESS", "I-ADDRESS", "B-CONTRACT_TYPE", "I-CONTRACT_TYPE",
    "B-SKILL", "I-SKILL", "B-DEGREE", "I-DEGREE", "B-EXPERIENCE", "I-EXPERIENCE",
    "B-SALARY", "I-SALARY", "B-BENEFIT", "I-BENEFIT", "B-START_DATE", "I-START_DATE",
    "B-DURATION", "I-DURATION", "B-CONTACT", "I-CONTACT", "B-EMAIL", "I-EMAIL",
    "B-PHONE", "I-PHONE", "B-TECH", "I-TECH", "B-INDUSTRY", "I-INDUSTRY",
    "B-LANGUAGE", "I-LANGUAGE", "B-LAW", "I-LAW", "O"  # Include "Outside" tag
]

# Create label mappings
def create_label_mapping():
    tag2id = {tag: idx for idx, tag in enumerate(CUSTOM_TAGS)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    return tag2id, id2tag

tag2id, id2tag = create_label_mapping()

# Prepare training data
def prepare_training_data(file_path):
    sentences = []
    sentence_tokens = []
    sentence_labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        headers = lines[0].strip().split(";")

        if headers != ["Token", "Label"]:
            raise ValueError("File does not have valid headers: 'Token' and 'Label'.")

        for line in lines[1:]:
            parts = line.strip().split(";")
            if len(parts) != 2:
                continue

            token, label = parts
            token = token.strip()
            label = label.strip()

            if not token or not label:
                if sentence_tokens:
                    sentences.append((sentence_tokens, sentence_labels))
                    sentence_tokens = []
                    sentence_labels = []
                continue

            if label not in tag2id:
                continue

            sentence_tokens.append(token)
            sentence_labels.append(tag2id[label])

        if sentence_tokens:
            sentences.append((sentence_tokens, sentence_labels))

    return sentences

training_sentences = prepare_training_data(training_data_file)

# Fine-tune with Hugging Face
def train_model(sentences):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(tag2id))

    def tokenize_and_align_labels(sentences):
        tokenized_inputs = tokenizer(
            [tokens for tokens, _ in sentences],
            truncation=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        aligned_labels = []

        for i, (tokens, labels) in enumerate(sentences):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_labels.append(
                [-100 if word_id is None else labels[word_id] for word_id in word_ids]
            )

        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    tokenized_data = tokenize_and_align_labels(sentences)

    dataset = Dataset.from_dict({
        "input_ids": tokenized_data["input_ids"].tolist(),
        "attention_mask": tokenized_data["attention_mask"].tolist(),
        "labels": tokenized_data["labels"]
    })

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    return model, tokenizer

model, tokenizer = train_model(training_sentences)

# NER analysis
def ner_analysis(input_file, model, tokenizer, output_file):
    job_data = pd.read_csv(input_file, delimiter=',', quotechar='"', encoding='utf-8')
    ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    results = []
    for _, row in job_data.iterrows():
        text = row["Job Description"]
        entities = ner(text)

        formatted_entities = []
        for entity in entities:
            formatted_entities.append({
                "text": entity["word"].replace("##", ""),
                "label": entity["entity_group"]
            })

        results.append({"Job Description": text, "Entities": formatted_entities})

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

ner_analysis(job_descriptions_file, model, tokenizer, output_file)
