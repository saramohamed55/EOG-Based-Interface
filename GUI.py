import time
import tkinter as tk

import Test


def handle_button_click(activity):
    new_window = tk.Toplevel(window)
    new_window.title("Activity Confirmation")
    set_window_style(new_window)

    # Create a label to display the chosen activity
    label = tk.Label(new_window, text=f"You have chosen to {activity}.", font=("Arial", 16))
    label.pack(padx=20, pady=20)

    # Create a back button to return to the previous window
    back_button = tk.Button(new_window, text="Back", width=10, height=2, bg="#000000", fg="#FFFFFF",
                            command=new_window.destroy)
    back_button.pack(pady=10)


def set_window_style(window):
    window.configure(bg=window_bg_color)


def update_button_color(button, color):
    button.configure(bg=color)


def show_next_prediction(predictions, index):

    if index >= len(predictions):
        return

    value = predictions[index]

    if value == "up":
        update_button_color(sleep_button, "#00FF00")  # Change the sleep button color to green
        window.after(1000, lambda: update_button_color(sleep_button, "#000000"))  # Revert sleep button color to black after 1 second
    elif value == "down":
        update_button_color(bathroom_button, "#0000FF")  # Change the bathroom button color to blue
        window.after(1000, lambda: update_button_color(bathroom_button, "#000000"))  # Revert bathroom button color to black after 1 second
    elif value == "left":
        update_button_color(eat_button, "#FFFF00")  # Change the eat button color to yellow
        window.after(1000, lambda: update_button_color(eat_button, "#000000"))  # Revert eat button color to black after 1 second
    elif value == "right":
        update_button_color(drink_button, "#FF0000")  # Change the drink button color to red
        window.after(1000, lambda: update_button_color(drink_button, "#000000"))  # Revert drink button color to black after 1 second
    else:
         print("blink")  # Print "blink" for invalid input
         window.after(1000, lambda: None)  
    window.after(2000, show_next_prediction, predictions, index + 1)


def reset_button_colors():
    update_button_color(sleep_button, "#000000")  # Reset the sleep button color
    update_button_color(bathroom_button, "#000000")  # Reset the bathroom button color
    update_button_color(eat_button, "#000000")  # Reset the eat button color
    update_button_color(drink_button, "#000000")  # Reset the drink button color


# Create the main application window
window = tk.Tk()
window.title("EOG Interface")
window.geometry("400x400")
window_bg_color = "#8B00FF"
window.configure(bg=window_bg_color)
# Create the activity selection buttons
button_padding = 20

# Sleep button
sleep_button = tk.Button(window, text="Sleep", width=10, height=2, bg="#000000", fg="#FFFFFF",
                         command=lambda: handle_button_click('sleep'))
sleep_button.grid(row=0, column=1, pady=button_padding)

# Eat button
eat_button = tk.Button(window, text="Eat", width=10, height=2, bg="#000000", fg="#FFFFFF",
                       command=lambda: handle_button_click('eat'))
eat_button.grid(row=1, column=0, pady=button_padding)

# Drink button
drink_button = tk.Button(window, text="Drink", width=10, height=2, bg="#000000", fg="#FFFFFF",
                         command=lambda: handle_button_click('drink'))
drink_button.grid(row=1, column=2, pady=button_padding)

# Bathroom button
bathroom_button = tk.Button(window, text="Bathroom", width=10, height=2, bg="#000000", fg="#FFFFFF",
                            command=lambda: handle_button_click('bathroom'))
bathroom_button.grid(row=2, column=1, pady=button_padding)

# Configure the grid weights
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(2, weight=1)
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(2, weight=1)

delay = 2000 * 10  # Delay for displaying 10 predictions
# Start the main application loop
window.after(0, show_next_prediction, Test.prediction, 0)
window.mainloop()
