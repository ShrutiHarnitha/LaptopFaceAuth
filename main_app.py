from ModelTransferLearning import ModelFineTuning
from PredictFace import predict_candidate
from face_registration import face_register

import tkinter as tk
from tkinter import simpledialog,messagebox
from datetime import datetime
import csv


class LaptopFaceAuthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LaptopFaceAuth")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)

        self.app_label = tk.Label(self.main_frame, text="LaptopFaceAuth", font=("Helvetica", 24))
        self.app_label.pack(pady=10)

        self.register_button = tk.Button(self.main_frame, text="Register New Candidate",bg="lightgrey", command=self.register_candidate)
        self.register_button.pack(pady=10)
 
        self.attendence_button = tk.Button(self.main_frame, text="Take Attendence",bg="lightgrey", command=self.mark_attendence)
        self.attendence_button.pack(pady=10)
  
    
    def register_candidate(self):
        candidate_name = simpledialog.askstring("Input", "Please enter the candidate's name:")
        if candidate_name:
            face_register(candidate_name,False)
            ModelFineTuning()

    def mark_attendence(self):
        face_register("",True)
        predicted_candidate = predict_candidate('test_face.jpg')
        print("Candidate:",predicted_candidate)
        messagebox.showinfo("Predicted Candidate", f"Candidate: {predicted_candidate}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('attendance.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, predicted_candidate])
        print(f"Written to CSV: {timestamp}, {predicted_candidate}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LaptopFaceAuthApp(root)
    root.mainloop()
