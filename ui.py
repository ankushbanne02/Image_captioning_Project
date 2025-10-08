import tkinter as tk
from tkinter import filedialog, messagebox
import re
from collections import defaultdict
import os
import threading
import time

# Backend mount folders
BASE_MOUNT = "C:/LOGS"
LOGS_FOLDER = os.path.join(BASE_MOUNT, "logs")
FAILED_FOLDER = os.path.join(BASE_MOUNT, "failed_logs")
JSON_FOLDER = os.path.join(BASE_MOUNT, "json")

os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(FAILED_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)


PARSE_TIMEOUT = 120        
JSON_WAIT_TIMEOUT = 150    
UPLOAD_WAIT_TIMEOUT = 170 

class LogAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📦 Parcel Sorting Log Analyzer")
        self.root.geometry("1000x600")
        self.root.configure(bg="#f4f4f9")

        self.logs = {}
        self.file_path = None
        self.all_lines = defaultdict(set)
        self.progress_labels = {}  # keep progress labels for each PLC card

        tk.Label(root, text="Parcel Sorting Log Analyzer",
                 font=("Helvetica", 18, "bold"), bg="#f4f4f9").pack(pady=15)

        controls = tk.Frame(root, bg="#f4f4f9")
        controls.pack(pady=10)
        tk.Button(controls, text="Upload .txt File", font=("Arial", 12),
                  bg="#0078d7", fg="white", relief="raised",
                  command=self.upload_file).pack(side=tk.LEFT, padx=10)

        self.info_label = tk.Label(root, text="No file uploaded yet.",
                                   bg="#f4f4f9", font=("Arial", 10, "italic"))
        self.info_label.pack()

        frame_container = tk.Frame(root, bg="#f4f4f9")
        frame_container.pack(fill="both", expand=True, padx=10, pady=10)

        canvas = tk.Canvas(frame_container, bg="#f4f4f9", highlightthickness=0)
        scrollbar_x = tk.Scrollbar(frame_container, orient="horizontal", command=canvas.xview)
        scrollbar_y = tk.Scrollbar(frame_container, orient="vertical", command=canvas.yview)
        self.display_frame = tk.Frame(canvas, bg="#f4f4f9")

        self.display_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.display_frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")

    # -------------------
    # Utility: update label safely from background threads
    # -------------------
    def safe_update_label(self, label, text, fg=None):
        def _update():
            label.config(text=text)
            if fg:
                label.config(fg=fg)
        self.root.after(0, _update)

    

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return

        self.file_path = file_path
        file_name = os.path.basename(file_path)
        self.info_label.config(text=f"Loaded file: {file_name}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        pattern = re.compile(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2},\d{3})")
        new_data = defaultdict(list)

        for line in lines:
            match = pattern.search(line)
            if match:
                date, _ = match.groups()
                plc_match = re.search(r'PLC-(\d+)', line)
                if not plc_match:
                    continue
                plc = plc_match.group(1)
                key = f"{date}_PLC{plc}"
                new_data[key].append(line.strip())

        if not new_data:
            messagebox.showerror("Error", "File does not appear to be in the expected format.")
            return

        for key, lines in new_data.items():
            for line in lines:
                self.all_lines[key].add(line)

        self.current_data = []
        for key, line_set in self.all_lines.items():
            times = re.findall(r"\d{2}:\d{2}:\d{2},\d{3}", "\n".join(line_set))
            if times:
                times_sorted = sorted(times)
                self.logs[key] = {"start": times_sorted[0], "end": times_sorted[-1]}
                self.current_data.append({"key": key, "start": times_sorted[0], "end": times_sorted[-1]})

        messagebox.showinfo("Upload Successful",
                            f"File uploaded successfully!\n\nFound {len(self.current_data)} unique (Date, PLC) pairs.")
        self.refresh_display()

   
   
    def refresh_display(self):
        for widget in self.display_frame.winfo_children():
            widget.destroy()

        col = 0
        for key, data in sorted(self.logs.items()):
            date, plc = key.split("_PLC")
            self._add_log_card(date, data["start"], data["end"], plc, col)
            col += 1

    def _add_log_card(self, date, start, end, plc, col):
        key = f"{date}_PLC{plc}"
        card = tk.Frame(self.display_frame, bg="white",
                        highlightbackground="#ccc", highlightthickness=1, bd=2)
        card.grid(row=0, column=col, padx=10, pady=10, sticky="n")

        tk.Label(card, text=f"📅 Date: {date} | PLC: {plc}", font=("Arial", 12, "bold"),
                 bg="white").pack(anchor="w", padx=10, pady=3)
        tk.Label(card, text=f"🕐 Start: {start}", font=("Arial", 11),
                 bg="white").pack(anchor="w", padx=25)
        tk.Label(card, text=f"🕓 End: {end}", font=("Arial", 11),
                 bg="white").pack(anchor="w", padx=25)

        tk.Button(card, text="⬇️ Download Logs",
                  bg="#0078d7", fg="white", font=("Arial", 10, "bold"),
                  command=lambda k=key: self.download_logs(k)).pack(anchor="e", padx=15, pady=5)

        upload_btn = tk.Button(card, text="⬆️ Upload to DB",
                               bg="#ffa500", fg="white", font=("Arial", 10, "bold"),
                               command=lambda k=key: self.upload_to_db(k))
        upload_btn.pack(anchor="e", padx=15, pady=5)

        progress_label = tk.Label(card, text="", bg="white", fg="gray", font=("Arial", 9, "italic"))
        progress_label.pack(anchor="w", padx=10, pady=5)
        self.progress_labels[key] = progress_label

    
    def upload_to_db(self, key):
        if key not in self.all_lines:
            messagebox.showerror("Error", f"No logs found for {key}")
            return

        date, plc = key.split("_PLC")
        temp_path = os.path.join(LOGS_FOLDER, f"{date}_PLC{plc}.txt")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                for line in sorted(self.all_lines[key]):
                    f.write(line + "\n")
        except Exception as e:
            messagebox.showerror("Error writing to backend logs folder", str(e))
            return

        label = self.progress_labels[key]
        
        threading.Thread(target=self.monitor_backend, args=(key, label), daemon=True).start()

    
    def monitor_backend(self, key, label):
        try:
            date, plc = key.split("_PLC")
            log_filename = f"{date}_PLC{plc}.txt"
            log_path = os.path.join(LOGS_FOLDER, log_filename)

            # 1) Immediately after writing
            self.safe_update_label(label, "📤 Started parsing...", "blue")

            # 2) Wait until file is removed from LOGS_FOLDER (parsing done)
            start_t = time.time()
            while os.path.exists(log_path) and (time.time() - start_t) < PARSE_TIMEOUT:
                time.sleep(0.5)

            if os.path.exists(log_path):
                # timed out / still present
                self.safe_update_label(label, "⚠ Parsing timeout — file still in logs folder", "orange")
                return

            # file removed => parsing completed
            self.safe_update_label(label, "✅ Completed parsing", "green")

            # 3) Wait for JSON to appear in JSON_FOLDER or FAILED in FAILED_FOLDER
            self.safe_update_label(label, "🔎 Waiting for JSON (backend)...", "gray")
            json_found = False
            start_t = time.time()
            while (time.time() - start_t) < JSON_WAIT_TIMEOUT:
                # If backend moved a failed file for this date -> show error immediately
                failed_files = os.listdir(FAILED_FOLDER)
                if any(date in fname for fname in failed_files):
                    self.safe_update_label(
                        label,
                        " Error occurred while uploading file. Moved to failed_logs — contact technical team.",
                        "red"
                    )
                    return

                
                json_files = os.listdir(JSON_FOLDER)
                if any(date in fname for fname in json_files):
                    json_found = True
                    break

                time.sleep(0.5)

            if not json_found:
                self.safe_update_label(label, "⚠ No JSON detected (timeout). Please check backend.", "orange")
                return

            # JSON created -> uploading to DB
            self.safe_update_label(label, "📄 JSON created — uploading to database...", "orange")

            # 4) Wait until JSON file is removed (upload done) OR failed file appears
            start_t = time.time()
            while (time.time() - start_t) < UPLOAD_WAIT_TIMEOUT:
                # If failed file appears at any time -> error
                failed_files = os.listdir(FAILED_FOLDER)
                if any(date in fname for fname in failed_files):
                    self.safe_update_label(
                        label,
                        " Error occurred while uploading file. Moved to failed_logs — contact to technical team.",
                        "red"
                    )
                    return

                # If JSON no longer present => upload finished successfully
                json_files = os.listdir(JSON_FOLDER)
                if not any(date in fname for fname in json_files):
                    self.safe_update_label(label, "✅ Upload successful!", "green")
                    return

                time.sleep(0.5)

            # If we reach here, upload took too long
            self.safe_update_label(label, "⚠ Upload timeout — check backend logs", "orange")

        except Exception as e:
            # Unexpected errors
            self.safe_update_label(label, f" Monitoring error: {e}", "red")
            print("monitor_backend exception:", e)

    
    
    def download_logs(self, key):
        if key not in self.all_lines:
            messagebox.showerror("Error", f"No logs found for {key}")
            return

        date, plc = key.split("_PLC")
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"{date}_PLC{plc}.txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if not save_path:
            return

        with open(save_path, "w", encoding="utf-8") as f:
            for line in sorted(self.all_lines[key]):
                f.write(line + "\n")

        messagebox.showinfo("Download Complete", f"Logs saved to:\n{save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LogAnalyzerApp(root)
    root.mainloop()
