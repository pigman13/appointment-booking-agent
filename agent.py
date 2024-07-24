import pandas as pd
import dateparser
import spacy
from datetime import datetime, timedelta
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pytz

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Load appointments
try:
    appointments_df = pd.read_csv('appointment-booking-agent/appointments.csv')
except FileNotFoundError:
    appointments_df = pd.DataFrame(columns=['Name', 'Date', 'Start', 'End'])

# Initialize text generation pipeline with H2O-Danube3-4B-Chat model
model_name = "h2oai/h2o-danube3-4b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

context = {}

def detect_name(user_input):
    doc = nlp(user_input)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def debug_log(message):
    print(f"[DEBUG] {message}")

def parse_time(user_input):
    try:
        settings = {'TIMEZONE': 'UTC', 'RETURN_AS_TIMEZONE_AWARE': True}
        datetime_object = dateparser.parse(user_input, settings=settings)
        if datetime_object:
            # Manually localize the datetime object
            utc = pytz.utc
            datetime_object = datetime_object.replace(tzinfo=utc)
            return datetime_object
    except Exception as e:
        debug_log(f"Error parsing date with dateparser: {e}")

    # Manual parsing for common phrases
    user_input = user_input.lower()
    now = datetime.now()
    base_time = None

    if "tomorrow" in user_input:
        base_time = now + timedelta(days=1)
    elif "today" in user_input:
        base_time = now

    if base_time:
        # Extracting time from input
        time_phrases = ["at ", "by "]
        for phrase in time_phrases:
            if phrase in user_input:
                time_part = user_input.split(phrase)[1]
                time_parsed = dateparser.parse(time_part)
                if time_parsed:
                    base_time = base_time.replace(hour=time_parsed.hour, minute=time_parsed.minute, second=time_parsed.second)
                    return base_time.replace(tzinfo=pytz.utc)

    return None

def parse_duration(user_input):
    nlp_user_input = nlp(user_input)
    for ent in nlp_user_input.ents:
        if ent.label_ == "TIME" or ent.label_ == "DURATION":
            try:
                return pd.to_timedelta(ent.text)
            except ValueError:
                pass
    return None

def check_availability(date, start_time, end_time):
    global appointments_df
    
    # Combine date with start_time and end_time
    start_time = pd.to_datetime(f"{date} {start_time}")
    end_time = pd.to_datetime(f"{date} {end_time}")
    
    for _, row in appointments_df.iterrows():
        if row['Date'] == date:  # Check for the same date
            existing_start = pd.to_datetime(f"{row['Date']} {row['Start']}")
            existing_end = pd.to_datetime(f"{row['Date']} {row['End']}")
            if not (end_time <= existing_start or start_time >= existing_end):
                return False
    return True

def book_appointment(name, date, start_time, end_time):
    global appointments_df
    new_appointment = pd.DataFrame([{'Name': name, 'Date': date, 'Start': start_time, 'End': end_time}])
    appointments_df = pd.concat([appointments_df, new_appointment], ignore_index=True)
    appointments_df.to_csv('appointment-booking-agent/appointments.csv', index=False)
    debug_log(f"Booked appointment: {new_appointment.to_dict(orient='records')}")
    print("Updated Appointments CSV:")
    print(appointments_df)

def generate_reply(messages):
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    response = pipe(
        prompt,
        return_full_text=False,
        max_new_tokens=32,
        temperature=1.0,
        top_p=0.5,
        do_sample=True,
        num_beams=1
    )
    return response[0]["generated_text"]

def cancel_booking(user_input):
    global appointments_df
    debug_log(f"User input for cancellation: {user_input}")

    name = detect_name(user_input)
    if not name:
        return "Please provide the name for the reservation you want to cancel."

    # Check if the name exists in the appointments
    if name.lower() in appointments_df['Name'].str.lower().values:
        appointments_df = appointments_df[appointments_df['Name'].str.lower() != name.lower()]
        appointments_df.reset_index(drop=True, inplace=True)
        appointments_df.to_csv('appointments.csv', index=False)
        debug_log(f"Cancelled appointment for: {name}")
        return f"The reservation under the name {name} has been cancelled."
    else:
        return f"No reservation found under the name {name}."

def handle_user_input(user_input, am_pm=None, duration=None):
    debug_log(f"User input: {user_input}, AM/PM: {am_pm}, Duration: {duration}")

    booking_keywords = ["book", "make a reservation", "make an appointment"]
    cancellation_keywords = ["cancel", "delete", "remove", "cancel reservation", "cancel appointment"]

    # Check for cancellation keywords first
    if any(keyword in user_input.lower() for keyword in cancellation_keywords):
        return cancel_booking(user_input)

    # Initialize booking context if booking keywords are detected
    if any(keyword in user_input.lower() for keyword in booking_keywords):
        if 'booking' not in context:
            context['booking'] = {'name': None, 'date': None, 'time': None, 'duration': None, 'am_pm': None}

        # Use the name detection function
        detected_name = detect_name(user_input)
        if detected_name:
            context['booking']['name'] = detected_name

        # Attempt to parse date and time from the initial user input
        parsed_date = parse_time(user_input)
        if parsed_date:
            context['booking']['date'] = parsed_date.strftime('%Y-%m-%d')
            parsed_time = parsed_date.time().strftime('%H:%M:%S')
            context['booking']['time'] = parsed_time

        parsed_duration = parse_duration(user_input)
        if parsed_duration:
            context['booking']['duration'] = parsed_duration

        if 'am' in user_input.lower() or 'pm' in user_input.lower():
            context['booking']['am_pm'] = 'PM' if 'pm' in user_input.lower() else 'AM'

    if 'booking' in context and context['booking']:
        if context['booking']['name'] is None:
            detected_name = detect_name(user_input)
            if detected_name:
                context['booking']['name'] = detected_name
            else:
                return "What's the name for the reservation?"

        if context['booking']['date'] is None:
            parsed_date = parse_time(user_input)
            if parsed_date:
                context['booking']['date'] = parsed_date.strftime('%Y-%m-%d')
            else:
                return "When would you like to make the appointment?"

        if context['booking']['time'] is None:
            parsed_time = parse_time(user_input)
            if parsed_time:
                context['booking']['time'] = parsed_time.strftime('%H:%M:%S')
            else:
                return "Please specify the time in 24-hour format or mention AM/PM."

        if context['booking']['am_pm'] is None:
            if 'am' in user_input.lower():
                context['booking']['am_pm'] = 'AM'
            elif 'pm' in user_input.lower():
                context['booking']['am_pm'] = 'PM'
            else:
                return "Is the time AM or PM?"

        if context['booking']['duration'] is None:
            parsed_duration = parse_duration(user_input)
            if parsed_duration:
                context['booking']['duration'] = parsed_duration
            else:
                return "How long will the appointment be?"

        # Once all information is gathered, finalize the booking
        date = context['booking']['date']
        start_time_str = context['booking']['time']
        start_time = datetime.strptime(start_time_str, '%H:%M:%S')

        # Correct handling of AM/PM
        if context['booking']['am_pm']:
            if context['booking']['am_pm'].lower() == 'pm' and start_time.hour < 12:
                start_time += timedelta(hours=12)
            elif context['booking']['am_pm'].lower() == 'am' and start_time.hour >= 12:
                start_time -= timedelta(hours=12)

        end_time = start_time + context['booking']['duration']
        start_time_str = start_time.strftime('%H:%M:%S')
        end_time_str = end_time.strftime('%H:%M:%S')

        if check_availability(date, start_time_str, end_time_str):
            book_appointment(context['booking']['name'], date, start_time_str, end_time_str)
            debug_log(f"Booking Info: Name = {context['booking']['name']}, Date = {date}, Time = {start_time_str}, Duration = {context['booking']['duration']}")
            context.pop('booking')
            return f'Your appointment is booked for {date} from {start_time_str} to {end_time_str}.'
        else:
            context.pop('booking')
            return "The slot for the provided time is already occupied. Please choose another time."

    # General assistant functionality
    messages = [{"role": "user", "content": user_input}]
    reply = generate_reply(messages)
    return reply