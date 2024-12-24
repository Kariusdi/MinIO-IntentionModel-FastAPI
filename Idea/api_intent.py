from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_intent_classifier(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    return tokenizer, model

def predict_intent(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    intent_mapping = ["Scholarship", "academic_calendar", "course", "general_question","student_activities"]
    return intent_mapping[predicted_class]

def handle_intent(intent, prompt):
    if intent in ["academic_calendar", "student_activities"]:
        # API Call
        Get_api_data = f"ข้อมูลที่ดึงมาจาก API สำหรับ {intent}"
        return f"ตอบคำถามจาก API: {Get_api_data}\nคำถาม: {prompt}"
    elif intent in ["Scholarship", "course", "general_question"]:
        #  RAG (Vector Database)
        Get_database_scope = f"{intent}_vector_database"
        return f"ตอบคำถามโดย RAG (ค้นหาจาก {Get_database_scope})\nคำถาม: {prompt}"
    else:
        # Default Case for Unclassified Intent
        return f"ตอบคำถามโดย RAG (ค้นหาจาก general_question_vector_database)\nคำถาม: {prompt}"
    

def main():
    model_path = "/home/s6410301021/MinIO-IntentionModel-FastAPI/Idea/results/best_model"

    tokenizer, model = load_intent_classifier(model_path)

    prompt = input("กรุณากรอกคำถามของคุณ: ")
    intent = predict_intent(prompt,tokenizer,model)
    print(f"[Intent]:{intent}")
    response = handle_intent(intent, prompt)
    print(f"[Response]: {response}")
    # user_prompts = [
    #     "รายละเอียดเกี่ยวกับเครื่องมือที่เน้นในการเรียนการสอนในหลักสูตรนี้คืออะไร?",
    #     "ช่วยบอกวัตถุประสงค์หลักในการจัดทำหลักสูตรนี้?",
    #     "รายวิชา (การออกแบบระบบฝังตัว) เน้นการเรียนรู้ในหัวข้ออะไร?",
    #     "ในปีการศึกษาที่ 2 ภาคการศึกษาที่ 3 มีวิชาเลือกใดที่นักศึกษาเรียนได้?"
    # ]

    # for prompt in user_prompts:
    #     # Step 1: Predict Intent
    #     intent = predict_intent(prompt, tokenizer, model)
    #     print(f"\n[Intent]: {intent}")

    #     # Step 2: Handle Intent
    #     response = handle_intent(intent, prompt)
    #     print(f"[Response]: {response}")

# Run the program
if __name__ == "__main__":
    main()




