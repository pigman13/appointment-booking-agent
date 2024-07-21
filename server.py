from scripts.agent import handle_user_input, process_booking, process_cancellation, generate_reply
from scripts.utils import load_appointments

def chat_with_agent():
    df = load_appointments()
    print("Welcome to the Appointment Booking Agent!")
    state = {'step': 'initial'}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response, state = handle_user_input(user_input, df, state)
        print(f"Agent: {response}")

if __name__ == "__main__":
    chat_with_agent()
