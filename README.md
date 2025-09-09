# LLM Fine-Tuning with Llama 3.1 🤖

This repository contains a Python script for **fine-tuning** a large language model (LLM), specifically the **Meta Llama 3.1 8B Instruct** model, on a custom dataset. The script uses the **Hugging Face `transformers` library** and leverages techniques like **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA** and **4-bit quantization** to make the process more efficient and accessible.

***

## 🚀 Key Features

* **Model Fine-Tuning**: Trains the Llama 3.1 model on a custom dataset of "dad jokes."
* **PEFT with LoRA**: Uses the **Low-Rank Adaptation (LoRA)** method to fine-tune the model efficiently by updating only a small number of parameters.
* **4-bit Quantization (QLoRA)**: Reduces the model's memory footprint by loading it in 4-bit precision, making it possible to run on consumer GPUs.
* **Custom Dataset Handling**: Loads a dataset from a local CSV file, mapping a "joke" column to the required "text" format for training.
* **Hugging Face Integration**: Seamlessly integrates with the Hugging Face ecosystem for model and tokenizer loading, and uses the `Trainer` class for the training loop.

***

## 💡 Motivation
An article I wrote providing the motivation for this project can be found here:

[<img width="500" height="412.6" alt="image" src="https://github.com/user-attachments/assets/41298454-7e96-4586-8b3b-b17a0f481e44" />](https://medium.com/@jhavera/dad-jokes-lora-and-my-daughter-a-misguided-technical-odyssey-1ed6d6dd908c)

[Dad Jokes, LoRA, and My Daughter: A Misguided Technical Odyssey](https://medium.com/@jhavera/dad-jokes-lora-and-my-daughter-a-misguided-technical-odyssey-1ed6d6dd908c)

***

## 🛠️ Requirements

Before running the script, make sure you have the necessary libraries installed.

```bash
pip install torch transformers datasets peft accelerate bitsandbytes
```
## 📝 How It Works

The script follows a standard fine-tuning workflow:

1.  **Load Dataset**: The `load_csv_dataset` function reads a CSV file (e.g., `data/dad_jokes.csv`), formats it, and prepares it for training.
2.  **Load Model and Tokenizer**:
    * The `Llama-3.1-8B-Instruct` model and its corresponding tokenizer are loaded from the Hugging Face Hub.
    * **4-bit quantization** is applied using `BitsAndBytesConfig` if a CUDA-enabled GPU is detected. This significantly reduces the model's memory usage.
3.  **Prepare for PEFT**:
    * A `LoraConfig` is defined to specify the LoRA parameters, such as rank (`r`) and alpha (`lora_alpha`).
    * The base model is prepared for **k-bit training** and wrapped with the PEFT model class to enable LoRA.
4.  **Tokenize Data**: The dataset text is converted into a numerical format (tokens) that the model can understand. This process includes padding and truncation to ensure consistent input lengths.
5.  **Set Up Trainer**:
    * The `TrainingArguments` class is used to configure training parameters like learning rate, number of epochs, batch size, and logging.
    * A `Trainer` instance is created, which handles the entire training process, including the forward and backward passes, optimization, and evaluation.
6.  **Train**: The `trainer.train()` method starts the fine-tuning process. The model learns to generate new "dad jokes" based on the provided dataset.

Once training is complete, the script will output a confirmation message. The fine-tuned adapter weights can then be saved and used for inference.
