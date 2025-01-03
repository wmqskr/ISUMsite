# -*- coding: utf-8 -*-
import codecs
import csv
import pickle
import Tkinter as tk
import tkFileDialog
import tkMessageBox
import Data_Process_Tools as Tools

def select_file():
    file_path = tkFileDialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        print("Selected file path:", file_path)
        Tools.process_file(file_path)
        # Close the window after processing the file
        root.destroy()

# Create the main window
root = tk.Tk()
root.title("Protein Sequence Prediction")

# Set window size
window_width = 512
window_height = 512
root.geometry("{}x{}".format(window_width, window_height))

# Create a button
button = tk.Button(root, text="Select the .fsa file", command=select_file, width=20, height=2)
button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Enter the main event loop
root.mainloop()

# Load the model
with open("new_ensemble.pkl", "rb") as model_file:
    ensemble_model = pickle.load(model_file)

# Read the test set
def read_csv(file_path):
    features = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            features.append([float(x) for x in row])
    return features

# Load the feature dataset
test_features = read_csv("./Data/Processed_Data/featured.csv")

# Make predictions
predictions = ensemble_model.predict(test_features)
probabilities = ensemble_model.predict_proba(test_features)

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.page_index = 0
        self.init_window()

    def init_window(self):
        # Window size
        self.master.title("Protein Sequence Prediction")
        self.master.geometry("{}x{}".format(window_width, window_height))
        # Size of the text display
        self.text_box = tk.Text(self.master, height=30, width=60)
        self.text_box.pack()

        # Previous page button
        self.prev_page_button = tk.Button(self.master, text="Previous Page", command=self.prev_page, padx=0, pady=5)
        self.prev_page_button.pack(side=tk.LEFT, padx=25, pady=5)

        # Next page button
        self.next_page_button = tk.Button(self.master, text="Next Page", command=self.next_page, padx=0, pady=5)
        self.next_page_button.pack(side=tk.LEFT, padx=25, pady=5)

        # Quit button
        self.quit_button = tk.Button(self.master, text="Quit", command=self.quit_program, padx=20, pady=5)
        self.quit_button.pack(side=tk.RIGHT, padx=25, pady=5)

        # Save button
        self.save_button = tk.Button(self.master, text="Save All", command=self.save_all_to_file, padx=20, pady=5)
        self.save_button.pack(side=tk.RIGHT, padx=25, pady=5)

        self.update_content()


    def update_content(self):
        self.text_box.delete(1.0, tk.END)
        start_index = self.page_index * 20
        end_index = min(start_index + 20, len(lines))
        for i in range(start_index, end_index):
            line_number = i + 1
            if line_number % 2 == 0:  # Even lines
                self.text_box.insert(tk.END, lines[i])

    def save_all_to_file(self):
        file_path = tkFileDialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with codecs.open(file_path, "w", encoding="utf-8") as file:
                for page_index in range(len(lines) // 20 + 1):
                    self.page_index = page_index
                    self.update_content()
                    content = self.text_box.get("1.0", "end")  # Get the content of the entire text box
                    file.write(content)
            tkMessageBox.showinfo("Saved successfully", "Data has been saved to file {}".format(file_path))

    def next_page(self):
        if (self.page_index + 1) * 20 < len(lines):
            self.page_index += 1
            self.update_content()

    def prev_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self.update_content()

    def quit_program(self):
        self.master.destroy()


def print_even_lines(filename):
    try:
        with open(filename, 'r') as file:
            global lines
            lines = file.readlines()
            protein_count = 1
            for i in range(1, len(lines), 2):  # Start at index 1, step by 2, i.e. even lines
                line = lines[i].strip() + '\n'
                line = "Protein {}: ".format(protein_count) + line  # Add "Protein X" before each protein sequence
                protein_count += 1
                # Print the probability information for the corresponding sample
                line += print_probabilities(i)
                lines[i] = line  # Replace the original data in the list
            root = tk.Tk()
            app = Application(master=root)
            app.mainloop()

    except IOError:
        print("File not found or unreadable")


def print_probabilities(start_index):
    start_index = float(start_index) / 2  # Use float division
    start_index = int(start_index)  # Convert the result back to an integer
    if probabilities[start_index][1] >= 0.5:
        return "{:.3f} -->positive sample\n\n".format(probabilities[start_index][1])
    else:
        return "{:.3f} -->negative sample\n\n".format(probabilities[start_index][1])

# Call the function to print the content and probability information of even lines
print_even_lines("./Data/Processed_Data/copy.txt")


# Output the statistical results
count_0 = sum(1 for prediction in predictions if prediction == 0)
count_1 = sum(1 for prediction in predictions if prediction == 1)
print("Number of negative samples in the prediction results: {}".format(count_0))
print("Number of positive samples in the prediction results: {}".format(count_1))