from datetime import datetime
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
from scripts.utils import load_appointments, save_appointments

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

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
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs['input_ids'], 
        max_length=150, 
        num_beams=5, 
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def handle_user_input(user_input, df, state):
    if state['step'] == 'initial':
        if "book" in user_input.lower():
            state['step'] = 'collect_name'
            return generate_reply("I can help you book an appointment. Please provide your name."), state
        elif "available" in user_input.lower():
            state['step'] = 'collect_date_for_availability'
            return generate_reply("Sure, please provide the date (YYYY-MM-DD), the start time (HH:MM:SS), and the end time (HH:MM:SS) to check availability."), state
        elif "cancel" in user_input.lower():
            state['step'] = 'collect_name_for_cancellation'
            return generate_reply("I can help you cancel an appointment. Please provide your name."), state
        else:
            return generate_reply(f"You said: {user_input}. How can I assist you further?"), state
    elif state['step'] == 'collect_name':
        state['name'] = user_input
        state['step'] = 'collect_date'
        return generate_reply("Please provide the date (YYYY-MM-DD) for your appointment."), state
    elif state['step'] == 'collect_date':
        state['date'] = user_input
        state['step'] = 'collect_start_time'
        return generate_reply("Please provide the start time (HH:MM:SS) for your appointment."), state
    elif state['step'] == 'collect_start_time':
        state['start_time'] = user_input
        state['step'] = 'collect_end_time'
        return generate_reply("Please provide the end time (HH:MM:SS) for your appointment."), state
    elif state['step'] == 'collect_end_time':
        state['end_time'] = user_input
        response = book_appointment(df, state['name'], state['date'], state['start_time'], state['end_time'])
        state['step'] = 'initial'
        return generate_reply(response), state
    elif state['step'] == 'collect_name_for_cancellation':
        state['name'] = user_input
        state['step'] = 'collect_date_for_cancellation'
        return generate_reply("Please provide the date (YYYY-MM-DD) of the appointment you want to cancel."), state
    elif state['step'] == 'collect_date_for_cancellation':
        state['date'] = user_input
        state['step'] = 'collect_start_time_for_cancellation'
        return generate_reply("Please provide the start time (HH:MM:SS) of the appointment you want to cancel."), state
    elif state['step'] == 'collect_start_time_for_cancellation':
        state['start_time'] = user_input
        state['step'] = 'collect_end_time_for_cancellation'
        return generate_reply("Please provide the end time (HH:MM:SS) of the appointment you want to cancel."), state
    elif state['step'] == 'collect_end_time_for_cancellation':
        state['end_time'] = user_input
        response = cancel_appointment(df, state['name'], state['date'], state['start_time'], state['end_time'])
        state['step'] = 'initial'
        return generate_reply(response), state
    elif state['step'] == 'collect_date_for_availability':
        state['date'] = user_input
        state['step'] = 'collect_start_time_for_availability'
        return generate_reply("Please provide the start time (HH:MM:SS) to check availability."), state
    elif state['step'] == 'collect_start_time_for_availability':
        state['start_time'] = user_input
        state['step'] = 'collect_end_time_for_availability'
        return generate_reply("Please provide the end time (HH:MM:SS) to check availability."), state
    elif state['step'] == 'collect_end_time_for_availability':
        state['end_time'] = user_input
        available = check_availability(df, state['date'], state['start_time'], state['end_time'])
        state['step'] = 'initial'
        return generate_reply("The time slot is available." if available else "The time slot is not available."), state
    return generate_reply(f"You said: {user_input}. How can I assist you further?"), state

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
