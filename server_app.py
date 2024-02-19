from flask import Flask, request
from model_load import Model
import torch
import json

# git lfs install
app = Flask(__name__)
models = Model()

@app.route('/')
def health():
    return 'App is working!'

@app.route('/generate', methods = ['POST', 'GET'])
def generate():
    data = request.get_json()
    promt = data["promt"]
    # return "Some text"
    model_input = models.eval_tokenizer(promt, return_tensors="pt").to("cuda")
    models.ft_model.eval()
    with torch.no_grad():
        answer = models.eval_tokenizer.decode(
            models.ft_model.generate(
                **model_input,
                max_new_tokens=2048,
                repetition_penalty=1.4
            )[0], skip_special_tokens=True)
    return answer


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="5000")
