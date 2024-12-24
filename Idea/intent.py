import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix



# import matplotlib.pyplot as plt
# import seaborn as sns


# 1. โหลดข้อมูล
def load_training_data(file_path):
    data = pd.read_csv(file_path)
    texts = data['text'].tolist()
    intents = data['intent'].tolist()
    return texts, intents

file_path = '/home/s6410301021/MinIO-IntentionModel-FastAPI/Idea/Data/train_data.csv'
texts, intents = load_training_data(file_path)

# 2. แบ่งข้อมูล Training และ Testing
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, intents, test_size=0.2, random_state=42
)

# 3. แปลง Intent เป็นตัวเลข
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# 4. โหลดโมเดลและ Tokenizer
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. Tokenize และเตรียม Dataset
def encode_texts(texts, tokenizer):
    return tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")

train_encodings = encode_texts(train_texts, tokenizer)
test_encodings = encode_texts(test_texts, tokenizer)

class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = IntentDataset(train_encodings, train_labels)
test_dataset = IntentDataset(test_encodings, test_labels)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir='/home/s6410301021/MinIO-IntentionModel-FastAPI/Idea/results',  # Directory for model outputs
    fp16=True,                         # Use mixed precision for faster training on GPU
    num_train_epochs=12,               # Increase epochs for better training on small data
    per_device_train_batch_size=16,    # Increase batch size for stability
    learning_rate=2e-5,                # Lower learning rate for more stable training
    weight_decay=0.01,                 # Reduce overfitting by penalizing large weights
    eval_strategy="epoch",             # กำหนดกลยุทธ์ในการประเมินผล โดยประเมินผลเมื่อจบแต่ละ epoch
    save_strategy="epoch",             # Save model after every epoch
    logging_dir='/home/s6410301021/MinIO-IntentionModel-FastAPI/Idea/logs',  # Directory for logs
    logging_steps=10,                  # Log training progress every 10 steps
    save_total_limit=1,                # Keep only the last 3 saved models to save space
    evaluation_strategy="epoch",       # Run evaluation every epoch
    load_best_model_at_end=True,       # Automatically load the best model at the end
    metric_for_best_model="eval_loss", # Use eval_loss to determine the best model
    greater_is_better=False,           # Lower eval_loss is better
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
# บันทึกโมเดลที่ดีที่สุด
best_model_path = "/home/s6410301021/MinIO-IntentionModel-FastAPI/Idea/results/best_model"
trainer.save_model(best_model_path)

# บันทึก Tokenizer
tokenizer.save_pretrained(best_model_path)

print(f"Best model and tokenizer saved at {best_model_path}")

# Prediction บน test dataset
test_predictions = trainer.predict(test_dataset).predictions.argmax(axis=1)

# Accuracy
accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {accuracy}")

report = classification_report(test_labels, test_predictions, target_names=label_encoder.classes_)
print(report)

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)
print("\nConfusion Matrix:")
print("Labels:", label_encoder.classes_)

# Format Confusion Matrix for better readability
for i, row in enumerate(conf_matrix):
    print(f"{label_encoder.classes_[i]}: {row}")


# 8. การทำนาย Intent
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    return label_encoder.inverse_transform([predicted_class])[0]

# Test
print("จากคำถามเรื่อง < กำหนดวันสุดท้ายสำหรับการชำระค่าเล่าเรียนของเทอมหน้าอยู่ในวันใด? > มี intent        คือ: " ,predict_intent("กำหนดวันสุดท้ายสำหรับการชำระค่าเล่าเรียนของเทอมหน้าอยู่ในวันใด?"))#Academic Calendar
print("จากคำถามเรื่อง < ถ้าตารางสอบของฉันเกิดการชนกัน ควรทำอย่างไรเพื่อแก้ไข? > มี intent               คือ: " ,predict_intent("ถ้าตารางสอบของฉันเกิดการชนกัน ควรทำอย่างไรเพื่อแก้ไข?"))#Academic Calendar
print("จากคำถามเรื่อง < หากนักศึกษาต้องการขอเลื่อนวันสอบ ต้องดำเนินการแจ้งล่วงหน้ากี่วัน? > มี intent         คือ: " ,predict_intent("หากนักศึกษาต้องการขอเลื่อนวันสอบ ต้องดำเนินการแจ้งล่วงหน้ากี่วัน?"))#Academic Calendar
print("จากคำถามเรื่อง < มีแนวทางใดสำหรับการลงทะเบียนเรียนล่วงหน้าในภาคการศึกษาถัดไป? > มี intent        คือ: " ,predict_intent("มีแนวทางใดสำหรับการลงทะเบียนเรียนล่วงหน้าในภาคการศึกษาถัดไป?"))#Academic Calendar
print("จากคำถามเรื่อง < ช่วงเวลาใดที่อนุญาตให้นักศึกษาเพิ่มวิชาหลังจากเปิดเรียนไปแล้ว? > มี intent         คือ: " ,predict_intent("ช่วงเวลาใดที่อนุญาตให้นักศึกษาเพิ่มรายวิชาหลังจากเปิดเรียนไปแล้ว?"))#Academic Calendar
print("---------------------------------------")
print("จากคำถามเรื่อง < ใครคือบุคคลที่ริเริ่มก่อตั้งสถาบันเทคโนโลยีจิตรลดา? > มี intent                      คือ: " ,predict_intent("ใครคือบุคคลที่ริเริ่มก่อตั้งสถาบันเทคโนโลยีจิตรลดา?"))#General Questions
print("จากคำถามเรื่อง < ความหมายของตราสัญลักษณ์ที่สถาบันเทคโนโลยีจิตรลดาใช้อยู่คืออะไร? > มี intent         คือ: " ,predict_intent("ความหมายของตราสัญลักษณ์ที่สถาบันเทคโนโลยีจิตรลดาใช้อยู่คืออะไร?"))#General Questions
print("จากคำถามเรื่อง < ชื่ออย่างเป็นทางการของมหาวิทยาลัยในภาษาอังกฤษคืออะไร? > มี intent               คือ: " ,predict_intent("ชื่ออย่างเป็นทางการของมหาวิทยาลัยในภาษาอังกฤษคืออะไร?"))#General Questions
print("จากคำถามเรื่อง < มหาวิทยาลัยได้รับการจัดอันดับในระดับประเทศหรือระดับโลกในปีนี้หรือไม่? > มี intent      คือ: " ,predict_intent("มหาวิทยาลัยได้รับการจัดอันดับในระดับประเทศหรือระดับโลกในปีนี้หรือไม่?"))#General Questions
print("จากคำถามเรื่อง < ปริญญาโทของมหาวิทยาลัยมีการเปิดสอนในสาขาวิชาใดบ้าง? > มี intent                คือ: " ,predict_intent("ปริญญาโทของมหาวิทยาลัยมีการเปิดสอนในสาขาวิชาใดบ้าง?"))#General Questions
print("---------------------------------------")
print("จากคำถามเรื่อง < Assembly ในสัปดาห์นี้จะพูดเกี่ยวกับหัวข้อใด? > มี intent                          คือ: " ,predict_intent("Assembly ในสัปดาห์นี้จะพูดเกี่ยวกับหัวข้อใด?"))#Student Activities
print("จากคำถามเรื่อง < ฉันสามารถเช็คตาราง Assembly ได้จากที่ไหน? > มี intent                        คือ: " ,predict_intent("ฉันสามารถเช็คตาราง Assembly ได้จากที่ไหน?"))#Student Activities
print("จากคำถามเรื่อง < หัวข้อของ Assembly ในครั้งถัดไปจะเกี่ยวข้องกับเรื่องอะไร? > มี intent              คือ: " ,predict_intent("หัวข้อของ Assembly ในครั้งถัดไปจะเกี่ยวข้องกับเรื่องอะไร?"))#Student Activities
print("จากคำถามเรื่อง < วันนี้มี assembly ไหมคะ? > มี intent                                        คือ: " ,predict_intent("วันนี้มี assembly ไหมคะ?"))#Student Activities
print("จากคำถามเรื่อง < Assembly ที่มีหัวข้อเกี่ยวกับเทคโนโลยี AI จะจัดขึ้นในปีนี้หรือเปล่า? > มี intent         คือ: " ,predict_intent("Assembly ที่มีหัวข้อเกี่ยวกับเทคโนโลยี AI จะจัดขึ้นในปีนี้หรือเปล่า?"))#Student Activities
print("---------------------------------------")
print("จากคำถามเรื่อง < ทุนการศึกษาประเภทต่างๆ มีข้อกำหนดและเงื่อนไขอะไรบ้าง? > มี intent             คือ: " ,predict_intent("ทุนการศึกษาประเภทต่างๆ มีข้อกำหนดและเงื่อนไขอะไรบ้าง?"))#Scholarship
print("จากคำถามเรื่อง < มีเอกสารหรือหลักฐานใดบ้างที่จำเป็นต้องใช้ในการสมัครขอทุน? > มี intent            คือ: " ,predict_intent("มีเอกสารหรือหลักฐานใดบ้างที่จำเป็นต้องใช้ในการสมัครขอทุน?"))#Scholarship
print("จากคำถามเรื่อง < ในการขอทุนการศึกษา นักศึกษาจำเป็นต้องเตรียมเอกสารอะไรบ้าง? > มี intent        คือ: " ,predict_intent("ผู้ในการขอทุนการศึกษา นักศึกษาจำเป็นต้องเตรียมเอกสารอะไรบ้าง?"))#Scholarship
print("จากคำถามเรื่อง < ช่องทางที่ใช้ประกาศรายชื่อผู้ได้รับทุนการศึกษาคืออะไร? > มี intent                 คือ: " ,predict_intent("ช่องทางที่ใช้ประกาศรายชื่อผู้ได้รับทุนการศึกษาคืออะไร?"))#Scholarship
print("จากคำถามเรื่อง < กระบวนการคัดเลือกทุนมีความโปร่งใสและยุติธรรมหรือไม่? > มี intent               คือ: " ,predict_intent("กระบวนการคัดเลือกทุนมีความโปร่งใสและยุติธรรมหรือไม่?"))#Scholarship
print("---------------------------------------")
print("จากคำถามเรื่อง < รายละเอียดเกี่ยวกับเครื่องมือที่เน้นในการเรียนการสอนในหลักสูตรนี้คืออะไร? > มี intent   คือ: " ,predict_intent("รายละเอียดเกี่ยวกับเครื่องมือที่เน้นในการเรียนการสอนในหลักสูตรนี้คืออะไร?"))#course
print("จากคำถามเรื่อง < ช่วยบอกวัตถุประสงค์หลักในการจัดทำหลักสูตรนี้? > มี intent                         คือ: " ,predict_intent("ช่วยบอกวัตถุประสงค์หลักในการจัดทำหลักสูตรนี้?"))#course
print("จากคำถามเรื่อง < รายวิชา (การออกแบบระบบฝังตัว) เน้นการเรียนรู้ในหัวข้ออะไร? > มี intent           คือ: " ,predict_intent("รายวิชา (การออกแบบระบบฝังตัว) เน้นการเรียนรู้ในหัวข้ออะไร?"))#course
print("จากคำถามเรื่อง < ในปีการศึกษาที่ 2 ภาคการศึกษาที่ 3 มีวิชาเลือกใดที่นักศึกษาเรียนได้? > มี intent        คือ: " ,predict_intent("ในปีการศึกษาที่ 2 ภาคการศึกษาที่ 3 มีวิชาเลือกใดที่นักศึกษาเรียนได้?"))#course
print("จากคำถามเรื่อง < วิชา ภาษาอังกฤษเพื่อการสื่อสาร มีเงื่อนไขในการลงทะเบียนอย่างไร? > มี intent        คือ: " ,predict_intent("วิชา ภาษาอังกฤษเพื่อการสื่อสาร มีเงื่อนไขในการลงทะเบียนอย่างไร?"))#course
