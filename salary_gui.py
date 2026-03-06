import tkinter as tk
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("salary_model.pkl","rb"))

# Predict function
def predict_salary():
    age = int(age_entry.get())
    exp = int(exp_entry.get())
    skill = int(skill_entry.get())
    edu = int(edu_entry.get())

    data = np.array([[age,exp,skill,edu]])

    result = model.predict(data)

    result_label.config(text="Predicted Salary: ₹" + str(int(result[0])))

# Create window
root = tk.Tk()
root.title("AI Salary Predictor")
root.geometry("400x400")

# Labels and Inputs
tk.Label(root,text="Age").pack()
age_entry = tk.Entry(root)
age_entry.pack()

tk.Label(root,text="Experience (years)").pack()
exp_entry = tk.Entry(root)
exp_entry.pack()

tk.Label(root,text="Skill (number)").pack()
skill_entry = tk.Entry(root)
skill_entry.pack()

tk.Label(root,text="Education (number)").pack()
edu_entry = tk.Entry(root)
edu_entry.pack()

# Button
predict_btn = tk.Button(root,text="Predict Salary",command=predict_salary)
predict_btn.pack()

# Result label
result_label = tk.Label(root,text="")
result_label.pack()

root.mainloop()