import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# Load the BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Input text
text = "Your input text goes here."

# Tokenize the input text
tokens = tokenizer(text, return_tensors='pt')

# Forward pass through the model to get the outputs
outputs = model(**tokens, output_attentions=True)

# Get the attention weights from the model's layers
try:
    attention = outputs.attentions  # This contains the attention weights
    layer_index = 0  # Change this to the desired layer index
    head_index = 0    # Change this to the desired head index

    # Extract attention weights for the specified layer and head if available
    if attention is not None:
        attention_weights = attention[layer_index][0][head_index].detach().numpy()

        # Plot the attention matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(attention_weights, cmap='viridis', interpolation='nearest')
        plt.title(f"Layer {layer_index + 1}, Head {head_index + 1} Attention Weights")
        plt.xlabel("Attention from")
        plt.ylabel("Attention to")
        plt.colorbar()
        plt.show()
    else:
        print("Attention weights are not available.")
except AttributeError as e:
    print(f"Error: {e}. The model might not provide attention weights.")
