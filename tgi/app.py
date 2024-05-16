from tgi import TGI
from huggingface_hub import InferenceClient
import time

llm = TGI(model_id="google/paligemma-3b-mix-224")
client = InferenceClient("http://localhost:3000")

while True:
    print("Waiting for the model to be ready...")
    try:
        time.sleep(5)
        generated = client.text_generation("What is Deep Learning?")
        break
    except Exception as e:
        print(e)

print("Model is ready!")

time.sleep(2)

# do a couple of inference requests
print("Generating text...")
generated = client.text_generation("Where is the capital of France?")
print(generated)

time.sleep(2)

generated = client.text_generation(
    "What can you tell me about the history of the United States?"
)
print(generated)

time.sleep(2)

generated = client.text_generation("What are the main characteristics of a cat?")
print(generated)

llm.close()
