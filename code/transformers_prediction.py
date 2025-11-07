# libraries
import torch 
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# model = "distilgpt2"
model = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model).to(device)
model.eval()

# prompt = "I think the Transformers movie is"

def predict_words(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    gen_kwargs = model.generate(
        **inputs, 
        # max_length=50,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(gen_kwargs[0], skip_special_tokens=True)

# Interface for display
view = gr.Interface(
    fn=predict_words,
    inputs=gr.Textbox(lines=4, placeholder="Start a sentencee, and the model" \
    "will complete it for you."),
    outputs="text",
    title="Text Generation with Transformers",
    description="This is a text generation app built using Transformers for Bootcamp 2025"
)

view.launch(server_name="0.0.0.0", server_port=7861, share=True)
