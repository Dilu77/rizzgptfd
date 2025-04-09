from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load model and tokenizer
MODEL_ID = os.getenv("MODEL_ID", "myownfd/rizzgpt-small")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Format input
    prompt = f"User: {user_message}\nRizzGPT:"
    
    # Generate response
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode and format response
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the RizzGPT part of the response
    try:
        response = generated_text.split("RizzGPT:")[1].strip()
    except IndexError:
        response = generated_text
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
