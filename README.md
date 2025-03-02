## 🧠 mindBERT - Mental Health Text Classification

![mindBERT UI](https://github.com/SYEDFAIZAN1987/mindBERT/blob/main/mindBERTUI.png)

### 🚀 Overview
**mindBERT** is a transformer-based model fine-tuned for **mental health text classification**. It utilizes state-of-the-art Natural Language Processing (NLP) techniques to detect various mental health conditions from textual data. The model is trained on the **Reddit Mental Health Dataset** and achieves high accuracy in detecting **stress, depression, bipolar disorder, personality disorder, and anxiety.**

👉 **Try it here:** [mindBERT UI](https://huggingface.co/spaces/DrSyedFaizan/mindBERT)

---

### 📌 Features
- **Fine-tuned BERT model** for **mental health classification**
- **Trained on real-world mental health text data** from Reddit
- **Achieves high accuracy on validation**
- **Optimized for inference and deployment**
- **Interactive UI available on Hugging Face Spaces**

---

### 📊 Training and Evaluation Results

#### **📉 Training Loss and Learning Rate Progression**
![Training Loss & Learning Rate](https://github.com/SYEDFAIZAN1987/mindBERT/blob/main/traininglossandlearningrate.png)

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|--------------|----------------|----------|
| 1     | 0.359400     | 0.285864       | 89.61%   |
| 2     | 0.210500     | 0.224632       | 92.03%   |
| 3     | 0.177800     | 0.217146       | 92.83%   |
| 4     | 0.089200     | 0.249640       | 93.23%   |
| 5     | 0.087600     | 0.282782       | 93.39%   |

#### **📈 Evaluation Metrics (Loss & Accuracy)**
![Evaluation Results](https://github.com/SYEDFAIZAN1987/mindBERT/blob/main/evalpics.png)

#### **🖼️ Confusion Matrix**
![Confusion Matrix](https://github.com/SYEDFAIZAN1987/mindBERT/blob/main/confusionmatrix.png)

#### **📊 Dataset Label Distribution**
![Dataset Labels](https://github.com/SYEDFAIZAN1987/mindBERT/blob/main/datasetlabelsbarh.png)

- **Final Accuracy:** 93.39%
- **Loss increased slightly after Epoch 4**, indicating potential early stopping.
- **Stable learning rate scheduling for optimal convergence.**

---

### 🔧 Model Architecture
mindBERT is built using **Hugging Face's `transformers` library**, leveraging **BERT-base** as a pre-trained backbone. The classification head consists of a dense layer followed by a softmax activation for multi-class classification.

### 🛠 Training Pipeline
The model was trained using **PyTorch** with the following configurations:

#### **🔧 Training Parameters**
```python
training_args = TrainingArguments(
    output_dir="./results",          # Output directory for results
    evaluation_strategy="epoch",     # Evaluate once per epoch
    save_strategy="epoch",          # Save model at the end of each epoch
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    num_train_epochs=5,              # Number of epochs
    weight_decay=0.01,               # Weight decay strength
    logging_dir="./logs",            # Directory for logging
    logging_steps=10,                # Log every 10 steps
    lr_scheduler_type="linear",      # Linear LR scheduler with warmup
    warmup_steps=500,                # Warmup steps for learning rate
    load_best_model_at_end=True,     # Load best model at end of training
    metric_for_best_model="eval_loss", # Monitor eval loss for best model
    save_total_limit=3,              # Limit checkpoints saved
    gradient_accumulation_steps=2,    # Simulate larger batch size
    report_to="wandb"                 # Report to Weights & Biases
)
```

---

### 🖥 How to Use
To use mindBERT for inference, follow these steps:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "DrSyedFaizan/mindBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sample text
text = "I feel so anxious and stressed all the time."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=1).item()

labels = ["Stress", "Depression", "Bipolar", "Personality Disorder", "Anxiety"]
print(f"Predicted Category: {labels[prediction]}")
```

---

### 📌 Future Improvements
- **Fine-tune with larger datasets** (e.g., CLPsych, eRisk)
- **Expand label categories** for broader mental health classification
- **Deploy as an API for real-world applications**

🔗 **Explore the app UI here:** [Hugging Face Spaces](https://huggingface.co/spaces/DrSyedFaizan/mindBERT)

🚀 **mindBERT - Advancing AI for Mental Health Research!**
