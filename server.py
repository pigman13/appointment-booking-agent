from scripts.agent import handle_user_input, process_booking, process_cancellation, generate_reply
from scripts.utils import load_appointments

def chat_with_agent():
    df = load_appointments()
    print("Welcome to the Appointment Booking Agent!")
    pending_booking = False
    pending_cancellation = False
    user_data = {}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        if pending_booking:
            # Collecting booking details
            if 'name' not in user_data:
                user_data['name'] = user_input
                print("Agent: Please provide the date (YYYY-MM-DD).")
            elif 'date' not in user_data:
                user_data['date'] = user_input
                print("Agent: Please provide the start time (HH:MM:SS).")
            elif 'start_time' not in user_data:
                user_data['start_time'] = user_input
                print("Agent: Please provide the end time (HH:MM:SS).")
            elif 'end_time' not in user_data:
                user_data['end_time'] = user_input
                response = process_booking(user_data['name'], user_data['date'], user_data['start_time'], user_data['end_time'], df)
                print(generate_reply(response))
                pending_booking = False
                user_data = {}
            continue
        
        if pending_cancellation:
            # Collecting cancellation details
            if 'name' not in user_data:
                user_data['name'] = user_input
                print("Agent: Please provide the date (YYYY-MM-DD).")
            elif 'date' not in user_data:
                user_data['date'] = user_input
                print("Agent: Please provide the start time (HH:MM:SS).")
            elif 'start_time' not in user_data:
                user_data['start_time'] = user_input
                print("Agent: Please provide the end time (HH:MM:SS).")
            elif 'end_time' not in user_data:
                user_data['end_time'] = user_input
                response = process_cancellation(user_data['name'], user_data['date'], user_data['start_time'], user_data['end_time'], df)
                print(generate_reply(response))
                pending_cancellation = False
                user_data = {}
            continue
        
        intent = handle_user_input(user_input, df)
        print(f"Agent: {intent}")
        
        if "please provide your name, date, start time, and end time" in intent.lower():
            if "cancel" in user_input.lower():
                pending_cancellation = True
            else:
                pending_booking = True

if __name__ == "__main__":
    chat_with_agent()
