# LLM Fine-Tuning with Llama 3.1 🤖

This repository contains a Python script for **fine-tuning** a large language model (LLM), specifically the **Meta Llama 3.1 8B Instruct** model, on a custom dataset. The script uses the **Hugging Face `transformers` library** and leverages techniques like **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA** and **4-bit quantization** to make the process more efficient and accessible.

---

## 🚀 Key Features

* **Model Fine-Tuning**: Trains the Llama 3.1 model on a custom dataset of "dad jokes."
* **PEFT with LoRA**: Uses the **Low-Rank Adaptation (LoRA)** method to fine-tune the model efficiently by updating only a small number of parameters.
* **4-bit Quantization (QLoRA)**: Reduces the model's memory footprint by loading it in 4-bit precision, making it possible to run on consumer GPUs.
* **Custom Dataset Handling**: Loads a dataset from a local CSV file, mapping a "joke" column to the required "text" format for training.
* **Hugging Face Integration**: Seamlessly integrates with the Hugging Face ecosystem for model and tokenizer loading, and uses the `Trainer` class for the training loop.

---

## 🛠️ Requirements

Before running the script, make sure you have the necessary libraries installed.

```bash
pip install torch transformers datasets peft accelerate bitsandbytes