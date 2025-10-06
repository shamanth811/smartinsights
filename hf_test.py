from huggingface_hub import InferenceClient
import os
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("Token present:", bool(token))
if token:
    client = InferenceClient(model="HuggingFaceH4/zephyr-7b-alpha", token=token)
    res = client.chat_completion(
        model="HuggingFaceH4/zephyr-7b-alpha",
        messages=[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Say hi"}],
        max_tokens=20
    )
    print("OK: ", res.choices[0].message["content"])
