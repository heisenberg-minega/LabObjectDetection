import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
import datetime
import csv
from PIL import Image, ImageTk  # NEW

# ----------------- Configurations --------------------
model = YOLO('yolov8n.pt')
lab_classes = ['laptop', 'mouse', 'keyboard', 'monitor', 'cpu', 'chair']
inventory = {item: 0 for item in lab_classes}
default_inventory = inventory.copy()
detected_log = []
current_object_counter = {item: 0 for item in lab_classes}
stop_detection_flag = False
frame_image = None  # NEW: used for holding reference to prevent garbage collection

# ----------------- Tkinter GUI -----------------------
root = tk.Tk()
root.title("Computer Lab Object Detection & Inventory")
root.geometry("1200x700")  # Wider window

# Main horizontal frame
main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

# Left: Controls panel
controls_frame = ttk.Frame(main_frame)
controls_frame.pack(side="left", fill="y", padx=10, pady=10)

selected_camera_index = tk.IntVar(value=0)

camera_frame = ttk.Frame(controls_frame)
camera_frame.pack(pady=5)
ttk.Label(camera_frame, text="Select Camera:", font=("Arial", 10)).pack(side="left", padx=5)

camera_options = [0, 1, 2]
camera_dropdown = ttk.Combobox(camera_frame, values=camera_options, width=5, state="readonly")
camera_dropdown.set(0)
camera_dropdown.pack(side="left", padx=5)

# Class checkboxes
checkbox_frame = ttk.LabelFrame(controls_frame, text="Select Objects to Detect")
checkbox_frame.pack(pady=10)
checkbox_vars = {}
for class_name in lab_classes:
    var = tk.BooleanVar(value=True)
    cb = ttk.Checkbutton(checkbox_frame, text=class_name.capitalize(), variable=var)
    cb.pack(anchor="w")
    checkbox_vars[class_name] = var

# Detected items list
ttk.Label(controls_frame, text="Detected Objects", font=("Arial", 12)).pack(pady=5)
listbox = tk.Listbox(controls_frame, width=40, height=8, font=("Arial", 10))
listbox.pack(pady=5)

# Inventory inputs
inventory_frame = ttk.LabelFrame(controls_frame, text="Inventory Summary (Editable)")
inventory_frame.pack(pady=10)
inventory_labels = {}
inventory_entries = {}
for item in lab_classes:
    frame = ttk.Frame(inventory_frame)
    frame.pack(fill="x", pady=2)
    lbl = ttk.Label(frame, text=f"{item.capitalize()}:", width=15, anchor="w", font=("Arial", 10))
    lbl.pack(side="left", padx=5)
    entry = ttk.Entry(frame, width=5)
    entry.insert(0, str(inventory[item]))
    entry.pack(side="left")
    inventory_labels[item] = lbl
    inventory_entries[item] = entry

# Buttons
ttk.Button(controls_frame, text="Start Detection", command=lambda: Thread(target=start_detection, daemon=True).start()).pack(pady=5)
stop_button = ttk.Button(controls_frame, text="Stop Detection", command=lambda: stop_detection(), state="disabled")
stop_button.pack(pady=5)
ttk.Button(controls_frame, text="Export Detection Log", command=lambda: export_csv()).pack(pady=5)
ttk.Button(controls_frame, text="Generate Inventory Report", command=lambda: generate_inventory_report()).pack(pady=5)
ttk.Button(controls_frame, text="Reset Inventory & Counts", command=lambda: reset_inventory_and_counters()).pack(pady=5)

# Right: Camera view panel
camera_view_frame = ttk.Frame(main_frame)
camera_view_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

camera_label = ttk.Label(camera_view_frame, text="Live Camera Feed", font=("Arial", 12))
camera_label.pack(pady=5)

camera_display = tk.Label(camera_view_frame)  # Label to hold video frame
camera_display.pack()

# ----------------- Functions ------------------------

def update_inventory_from_entries():
    for item, entry in inventory_entries.items():
        try:
            val = int(entry.get())
            if val < 0:
                raise ValueError
            inventory[item] = val
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a valid positive number for '{item}'.")
            return False
    return True

def update_inventory_display():
    for item, lbl in inventory_labels.items():
        expected = inventory[item]
        detected = current_object_counter.get(item, 0)
        lbl.config(text=f"{item.capitalize()}: {detected} / {expected}")
        lbl.config(foreground="green" if detected >= expected else "orange")

def reset_inventory_and_counters():
    if messagebox.askyesno("Confirm Reset", "Reset inventory and detection counts?"):
        for item in lab_classes:
            inventory_entries[item].delete(0, tk.END)
            inventory_entries[item].insert(0, str(default_inventory[item]))
            inventory[item] = default_inventory[item]
            current_object_counter[item] = 0
        update_inventory_display()
        listbox.delete(0, tk.END)
        messagebox.showinfo("Reset Done", "Inventory and counts reset.")

def export_csv():
    if not detected_log:
        messagebox.showinfo("No Data", "No detections to export.")
        return
    filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Confidence', 'Timestamp'])
        writer.writerows(detected_log)
    messagebox.showinfo("Exported", f"Saved as {filename}")

def generate_inventory_report():
    if not any(current_object_counter.values()):
        messagebox.showinfo("No Data", "No objects detected.")
        return
    report_data = []
    missing_items = []
    for item in lab_classes:
        expected = inventory[item]
        detected = current_object_counter.get(item, 0)
        if detected < expected:
            missing_items.append(f"{item.capitalize()}: Missing {expected - detected}")
        report_data.append([item, expected, detected])
    filename = f"inventory_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Item', 'Expected Count', 'Detected Count'])
        writer.writerows(report_data)
    if missing_items:
        messagebox.showwarning("Report", "⚠️ Missing:\n" + "\n".join(missing_items))
    else:
        messagebox.showinfo("Report", "✅ All items accounted for!")

def stop_detection():
    global stop_detection_flag
    stop_detection_flag = True
    stop_button.config(state="disabled")

def start_detection():
    global stop_detection_flag
    if not update_inventory_from_entries():
        return
    stop_button.config(state="normal")
    stop_detection_flag = False
    selected_cam = int(camera_dropdown.get())
    Thread(target=lambda: run_detection(selected_cam), daemon=True).start()

def run_detection(camera_index):
    global stop_detection_flag, frame_image
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened() and not stop_detection_flag:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result = results[0]
        boxes = result.boxes
        names = model.names

        selected_classes = [k for k, v in checkbox_vars.items() if v.get()]
        frame_counter = {item: 0 for item in lab_classes}
        current_detections = []

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]
            if class_name == "tv":
                class_name = "monitor"
            elif class_name == "microwave":
                class_name = "cpu"

            if class_name not in selected_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            detected_log.append([class_name, f"{conf:.2f}", timestamp])
            current_detections.append(f"{class_name} ({conf:.2f})")
            frame_counter[class_name] += 1

        for item in lab_classes:
            if frame_counter[item] > current_object_counter[item]:
                current_object_counter[item] = frame_counter[item]

        update_inventory_display()

        listbox.delete(0, tk.END)
        for item in current_detections:
            listbox.insert(tk.END, item)

        # Show frame in GUI
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        frame_image = ImageTk.PhotoImage(img_pil)
        camera_display.config(image=frame_image)

        root.update_idletasks()
        root.update()

    cap.release()
    camera_display.config(image="")
    stop_button.config(state="disabled")

# ----------------- Main ------------------------
root.mainloop()
