# Fine-tuned LLaMA 2 Model

This repository contains a **fine-tuned version of LLaMA 2-7B** for specialized tasks. The model has been trained using Hugging Face’s `transformers` and `PEFT` libraries to adapt LLaMA 2 to a domain-specific dataset.

---

## 📌 Model Overview

* **Base Model**: LLaMA 2 (7B / 13B / specify exact)
* **Fine-tuning Method**: \LoRA / PEFT 
* **Dataset**: \legal Dataset - https://zenodo.org/records/7152317#.Yz6mJ9JByC0
* **Objective**: \to simplify legal documents for non-experts

---

## 📂 Repository Structure

```
finetunedModel/
 └── checkpoint-20/
     ├── adapter_config.json
     ├── adapter_model.safetensors
     ├── chat_template.jinja
     ├── special_tokens_map.json
     ├── tokenizer.model
     ├── tokenizer_config.json
     ├── training_args.bin
     ├── optimizer.pt
     ├── mg_state.pth
     ├── trainer_state.json
     ├── tokenizer_config.json
     ├── README.md
```

* **adapter\_model.safetensors** → Fine-tuned model weights
* **adapter\_config.json** → Configuration for LoRA/PEFT
* **tokenizer.model** & **tokenizer\_config.json** → Tokenizer files
* **chat\_template.jinja** → Prompt formatting for chat-based interaction

---

## 🚀 How to Use

### Installation

```bash
pip install transformers peft accelerate safetensors
```

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base LLaMA 2 model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "path/to/finetunedModel/checkpoint-20")

# Generate text
input_text = "Explain the importance of AI in healthcare."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## ⚙️ Training Details

* **Training framework**: Hugging Face `transformers` + `PEFT`
* **Epochs / Steps**: \[20 checkpoints]
* **Batch Size**: \[2]
* **Optimizer**: AdamW
---

## 📜 License

This model is a fine-tuned version of **LLaMA 2**, which is licensed under the [Meta LLaMA 2 License](https://ai.meta.com/llama/license/).
Please ensure compliance with the base model’s license terms.

---

## 🙌 Acknowledgements

* [Meta AI](https://ai.meta.com/llama/) for LLaMA 2
* [Hugging Face](https://huggingface.co/) for `transformers` and `PEFT`
* \[NANDAGOPALNG] for fine-tuning and repository maintenance

