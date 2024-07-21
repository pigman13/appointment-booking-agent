from datetime import datetime
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.utils import load_appointments, save_appointments

# Load distilgpt2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

def check_availability(df, date, start_time, end_time):
    booked_slots = df[(df['Date'] == date) & 
                      ((df['Start'] < end_time) & (df['End'] > start_time))]
    return booked_slots.empty

def book_appointment(df, name, date, start_time, end_time):
    if check_availability(df, date, start_time, end_time):
        new_appointment = {'Name': name, 'Date': date, 'Start': start_time, 'End': end_time}
        df = df.append(new_appointment, ignore_index=True)
        save_appointments(df)
        return "Appointment booked successfully!"
    else:
        return "The time slot is already booked."

def cancel_appointment(df, name, date, start_time, end_time):
    mask = (df['Name'] == name) & (df['Date'] == date) & (df['Start'] == start_time) & (df['End'] == end_time)
    if not df[mask].empty:
        df = df[~mask]
        save_appointments(df)
        return "Appointment canceled successfully!"
    else:
        return "No matching appointment found to cancel."

def generate_reply(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=50, 
        do_sample=True, 
        temperature=0.7, 
        top_p=0.9, 
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def handle_user_input(user_input, df):
    if "book" in user_input.lower():
        return generate_reply("I can help you book an appointment. Please provide your name, the date (YYYY-MM-DD), the start time (HH:MM:SS), and the end time (HH:MM:SS).")
    elif "available" in user_input.lower():
        return generate_reply("Sure, please provide the date (YYYY-MM-DD), the start time (HH:MM:SS), and the end time (HH:MM:SS) to check availability.")
    elif "cancel" in user_input.lower():
        return generate_reply("I can help you cancel an appointment. Please provide your name, the date (YYYY-MM-DD), the start time (HH:MM:SS), and the end time (HH:MM:SS).")
    return generate_reply(f"You said: {user_input}. How can I assist you further?")

def process_booking(name, date, start_time, end_time, df):
    date = datetime.strptime(date, "%Y-%m-%d").date()
    start_time = datetime.strptime(start_time, "%H:%M:%S").time()
    end_time = datetime.strptime(end_time, "%H:%M:%S").time()
    return book_appointment(df, name, date, start_time, end_time)

def process_cancellation(name, date, start_time, end_time, df):
    date = datetime.strptime(date, "%Y-%m-%d").date()
    start_time = datetime.strptime(start_time, "%H:%M:%S").time()
    end_time = datetime.strptime(end_time, "%H:%M:%S").time()
    return cancel_appointment(df, name, date, start_time, end_time)
