from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from model_load import Model

app = FastAPI()

class InputData(BaseModel):
    prompt: str

@app.get('/')
def health():
    return {'message': 'App is working!'}

@app.post('/generate')
async def generate(data: InputData):
    prompt = data.prompt
    # You can perform some processing here based on the prompt
    generated_text = "Some text"
    return {'generated_text': generated_text}
    models = Model()
    model_input = models.eval_tokenizer(promt, return_tensors="pt").to("cuda")
    models.ft_model.eval()
    with torch.no_grad():
        answer = models.eval_tokenizer.decode(
            models.ft_model.generate(
                **model_input,
                max_new_tokens=2048,
                repetition_penalty=1.4
            )[0], skip_special_tokens=True)
    return {'generated_text': answer}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
