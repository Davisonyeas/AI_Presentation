# libraries
# !pip install gradio
# !pip install shap
import torch
import gradio as gr
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# loading pretrained senetiment classifier
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# clasification and prediction
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][1].item()
        label = "Positive" if confidence > 0.5 else "Negative"
    return f"Sentiment: {label} (Confidence: {confidence:.2f})"

# Interface for display
view = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter a sentnece here."),
    outputs="text",
    title="Sentiment Analyses with Transformer",
    description="This is a sentiment analysis app built using Transformers for Bootcamp 2025"
)

if __name__ === "__main__":
    view.launch(server_name="0.0.0.0", server_port=7860, share=True)
