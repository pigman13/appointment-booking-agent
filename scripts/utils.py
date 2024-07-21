import pandas as pd


filepath = 'appointment-booking-agent/data/appointments.csv'

def load_appointments(filepath=filepath):
    return pd.read_csv(filepath)

def save_appointments(df, filepath=filepath):
    df.to_csv(filepath, index=False)
