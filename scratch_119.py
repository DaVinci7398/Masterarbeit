import os
import pandas as pd
import nltk
from sklearn_crfsuite import CRF, metrics
from sklearn.model_selection import train_test_split

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Datei-Pfade
training_data_file = "/Users/vincentgruben/documents/Masterarbeit/bio_dataset15.csv"
job_descriptions_file = "/Users/vincentgruben/documents/Masterarbeit/test_file2.csv"
output_file = "/Users/vincentgruben/documents/Masterarbeit/NER_dataset_test.csv"

# Funktion zur Segmentierung
def load_and_segment_data(file_path, max_tokens_per_sentence=50):
    """
    Läd den Datensatz und segmentiert ihn in kleinere Sätze.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Spaltenüberschriften prüfen
    headers = lines[0].strip().split(";")
    if headers != ["Token", "Label"]:
        raise ValueError("Datensatz benötigt 'Token' und 'Label' als Spaltenüberschriften.")

    # Daten sammeln
    tokens = []
    labels = []

    for line in lines[1:]:
        parts = line.strip().split(";")
        if len(parts) != 2:
            continue  # Überspringe ungültige Zeilen
        token, label = parts
        tokens.append(token.strip())
        labels.append(label.strip())

    # Segmentierung in kleinere Sätze
    sentences = []
    sentence_tokens = []
    sentence_labels = []

    for i, (token, label) in enumerate(zip(tokens, labels)):
        sentence_tokens.append(token)
        sentence_labels.append(label)

        # Segmentiere basierend auf Tokenanzahl oder Satzgrenze
        if len(sentence_tokens) >= max_tokens_per_sentence or token in [".", "!", "?"]:
            sentences.append((sentence_tokens, sentence_labels))
            sentence_tokens = []
            sentence_labels = []

    # Letzten Satz hinzufügen
    if sentence_tokens:
        sentences.append((sentence_tokens, sentence_labels))

    print(f"Anzahl der segmentierten Sätze: {len(sentences)}")
    return sentences

# Features extrahieren
def word2features(sentence, i):
    """
    Extrahiert Features für jedes Token in einem Satz.
    """
    word = sentence[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sentence[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.istitle()': word1.istitle(),
        })
    else:
        features['BOS'] = True  # Beginning of Sentence

    if i < len(sentence) - 1:
        word1 = sentence[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.istitle()': word1.istitle(),
        })
    else:
        features['EOS'] = True  # End of Sentence

    return features

def sentence2features(sentence):
    return [word2features(sentence, i) for i in range(len(sentence))]

def sentence2labels(sentence):
    return [label for token, label in sentence]

def sentence2tokens(sentence):
    return [token for token, label in sentence]

# Trainingsdaten laden und segmentieren
training_sentences = load_and_segment_data(training_data_file)

# Trainingsdaten in Features umwandeln
X = [sentence2features(list(zip(tokens, labels))) for tokens, labels in training_sentences]
y = [labels for tokens, labels in training_sentences]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CRF-Modell trainieren
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Vorhersagen
y_pred = crf.predict(X_test)

# Evaluieren
labels = list(crf.classes_)
labels.remove("O")  # Ignoriere "Outside"-Label
print(metrics.flat_classification_report(y_test, y_pred, labels=labels))

# NER-Analyse
def ner_analysis(file_path, crf, output_file):
    """
    Wendet das NER-Modell auf neue Daten an und speichert die Ergebnisse.
    """
    job_data = pd.read_csv(file_path)
    results = []

    for _, row in job_data.iterrows():
        text = row["Job Description"].split()
        features = [word2features(list(zip(text, ["O"] * len(text))), i) for i in range(len(text))]
        prediction = crf.predict_single(features)
        entities = [{"token": token, "label": label} for token, label in zip(text, prediction) if label != "O"]
        results.append({"Job Description": " ".join(text), "Entities": entities})

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

ner_analysis(job_descriptions_file, crf, output_file)
