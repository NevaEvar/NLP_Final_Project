"""A small RoBERTa + dialogue-GRU classifier for NPS Chat."""

import random
from collections import Counter

import torch
from nltk.corpus import nps_chat
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "distilroberta-base"
MAX_LENGTH = 48
CHUNK_SIZE = 32


def ExamplesFromFiles(fileIds):
    """Return each chat file as a list of (utterance, dialogue-act) pairs."""
    dialogues = []
    for fileId in fileIds:
        posts = nps_chat.xml_posts(fileId)
        dialogue = []
        for post in posts:
            if post.text and post.text.strip():
                dialogue.append((post.text.strip(), post.attrib["class"]))
        if dialogue:
            dialogues.append(dialogue)
    return dialogues


def LoadTextEncoder(modelName=MODEL_NAME, device=None):
    """Load a frozen RoBERTa encoder and its tokenizer."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(modelName)
    encoder = AutoModel.from_pretrained(modelName).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    return tokenizer, encoder, device


def EncodeDialogues(dialogues, tokenizer, encoder, device, batchSize=64):
    """Convert utterances to frozen, mean-pooled RoBERTa vectors."""
    texts = [text for dialogue in dialogues for text, _ in dialogue]
    if not texts:
        raise ValueError("At least one non-empty dialogue is required.")

    encodedBatches = []
    with torch.no_grad():
        for start in range(0, len(texts), batchSize):
            inputs = tokenizer(
                texts[start : start + batchSize],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            encodedBatches.append(vectors.cpu())

    allVectors = torch.cat(encodedBatches)
    features = []
    offset = 0
    for dialogue in dialogues:
        end = offset + len(dialogue)
        features.append(allVectors[offset:end])
        offset = end

    return features


def CreateDialogueRNN(inputSize, hiddenSize, classCount):
    """Create a one-layer GRU that carries context through a chat."""
    contextRNN = torch.nn.GRU(
        input_size=inputSize,
        hidden_size=hiddenSize,
        batch_first=True,
    )
    outputLayer = torch.nn.Linear(hiddenSize, classCount)
    return contextRNN, outputLayer


def TrainClassifier(
    trainFeatures,
    trainLabels,
    classCount,
    device,
    epochs=8,
    hiddenSize=128,
    learningRate=0.001,
):
    """Train the GRU and output layer on short consecutive chat chunks."""
    if len(trainFeatures) != len(trainLabels) or not trainFeatures:
        raise ValueError("Features and labels must contain the same non-zero dialogues.")

    contextRNN, outputLayer = CreateDialogueRNN(
        trainFeatures[0].shape[1], hiddenSize, classCount
    )
    contextRNN.to(device)
    outputLayer.to(device)

    chunks = []
    labelCounts = Counter()
    for features, labels in zip(trainFeatures, trainLabels):
        if len(features) != len(labels):
            raise ValueError("Each dialogue must have one label per utterance.")
        labelCounts.update(labels)
        for start in range(0, len(labels), CHUNK_SIZE):
            end = start + CHUNK_SIZE
            chunks.append((features[start:end], labels[start:end]))

    classWeights = torch.tensor(
        [
            (max(labelCounts.values()) / max(labelCounts.get(label, 0), 1)) ** 0.5
            for label in range(classCount)
        ],
        dtype=torch.float32,
        device=device,
    )
    classWeights /= classWeights.mean()
    lossFunction = torch.nn.CrossEntropyLoss(weight=classWeights)
    optimizer = torch.optim.AdamW(
        list(contextRNN.parameters()) + list(outputLayer.parameters()),
        lr=learningRate,
    )

    randomGenerator = random.Random(42)
    for _ in range(epochs):
        randomGenerator.shuffle(chunks)
        contextRNN.train()
        outputLayer.train()
        for features, labels in chunks:
            inputs = features.unsqueeze(0).to(device)
            targets = torch.tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            context, _ = contextRNN(inputs)
            predictions = outputLayer(context.squeeze(0))
            loss = lossFunction(predictions, targets)
            loss.backward()
            optimizer.step()

    contextRNN.eval()
    outputLayer.eval()
    return contextRNN, outputLayer


def PredictDialogueActs(
    features,
    contextRNN,
    outputLayer,
    labelNames,
    device,
):
    """Predict one dialogue at a time, resetting context at each chunk."""
    predictedDialogues = []
    with torch.no_grad():
        for dialogueFeatures in features:
            predictedLabels = []
            for start in range(0, len(dialogueFeatures), CHUNK_SIZE):
                inputs = dialogueFeatures[start : start + CHUNK_SIZE]
                context, _ = contextRNN(inputs.unsqueeze(0).to(device))
                scores = outputLayer(context.squeeze(0))
                labelIds = scores.argmax(dim=1).cpu().tolist()
                predictedLabels.extend(labelNames[labelId] for labelId in labelIds)
            predictedDialogues.append(predictedLabels)

    return predictedDialogues
