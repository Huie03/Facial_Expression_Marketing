import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import threading
import queue

# --- Configuration & Data ---
PRODUCT_DATA = {
    "Stressed": [
        {"n": "Herbal Tea", "p": "RM 15", "i": "product_images/herbal_tea.png"},
        {"n": "Lavender Oil", "p": "RM 25", "i": "product_images/lavender_oil.png"},
        {"n": "Puzzle Game", "p": "RM 40", "i": "product_images/puzzle_game.png"}
    ],
    "Calm": [
        {"n": "Wireless Headphones", "p": "RM 199", "i": "product_images/wireless_headphones.png"},
        {"n": "New Sneakers", "p": "RM 250", "i": "product_images/sneakers.png"},
        {"n": "Indoor Plants", "p": "RM 35", "i": "product_images/indoor_plants.png"}
    ]
}

# --- AI Model Loading ---
try:
    model = tf.keras.models.load_model("stress_detector_cnn.h5", compile=False)
except Exception as e:
    print(f"Model Error: {e}")
    model = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
history = {"Calm": [], "Stressed": []}
report_data = [] 

class EmotionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Emotion Detection Marketing")
        self.window.geometry("1400x850")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.running = True
        self.frame_count = 0
        self.current_preds = [0.5, 0.5]
        self.result_queue = queue.Queue()
        
        self.stable_emotion = "Calm"
        self.emotion_timer = 0
        self.STABILITY_THRESHOLD = 5
        
        self.image_cache = {}
        self.preload_assets()
        
        self.colors = {"bg": "#F3F4F6", "sidebar": "#111827", "card": "#FFFFFF", "text": "#111827", "subtext": "#6B7280", "border": "#E5E7EB"}
        self.CLR_STRESS, self.CLR_CALM, self.CLR_ACCENT = "#EF4444", "#10B981", "#6366F1"

        self.setup_ui()
        self.create_recommendation_panels()
        
        self.cap = cv2.VideoCapture(0)
        self.update_app()

    def preload_assets(self):
        for state in PRODUCT_DATA:
            for p in PRODUCT_DATA[state]:
                if os.path.exists(p["i"]):
                    img = Image.open(p["i"]).resize((65, 65), Image.Resampling.LANCZOS)
                    self.image_cache[p["i"]] = ImageTk.PhotoImage(img)

    def setup_ui(self):
        for widget in self.window.winfo_children(): widget.destroy()
        self.window.configure(bg=self.colors["bg"])
        
        self.sidebar = tk.Frame(self.window, bg=self.colors["sidebar"], width=240)
        self.sidebar.pack(side="left", fill="y"); self.sidebar.pack_propagate(False)
        tk.Label(self.sidebar, text="RETAIL INSIGHTS", font=("Segoe UI", 20, "bold"), bg=self.sidebar['bg'], fg="white").pack(pady=40)

        self.btn_download = tk.Button(self.sidebar, text="DOWNLOAD CSV", command=self.download_report,
            font=("Segoe UI", 10, "bold"), bg=self.CLR_ACCENT, fg="white", cursor="hand2", relief="flat", pady=12)
        self.btn_download.pack(side="bottom", fill="x", padx=20, pady=20)

        self.btn_reset = tk.Button(self.sidebar, text="RESET SESSION", command=self.reset_session,
            font=("Segoe UI", 10, "bold"), bg="#4B5563", fg="white", cursor="hand2", relief="flat", pady=12)
        self.btn_reset.pack(side="bottom", fill="x", padx=20, pady=0)

        self.btn_summary = tk.Button(self.sidebar, text="VIEW SUMMARY", command=self.view_summary,
            font=("Segoe UI", 10, "bold"), bg="#374151", fg="white", cursor="hand2", relief="flat", pady=12)
        self.btn_summary.pack(side="bottom", fill="x", padx=20, pady=10)

        content = tk.Frame(self.window, bg=self.colors["bg"])
        content.pack(side="right", fill="both", expand=True, padx=30, pady=20)
        
        stats_f = tk.Frame(content, bg=self.colors["bg"])
        stats_f.pack(fill="x", pady=(0, 20))
        self.stress_lbl = self.create_stat(stats_f, "LIVE STRESS LEVEL", "0%", 0)
        self.status_lbl = self.create_stat(stats_f, "AI STATUS", "Tracking", 1)
        
        grid = tk.Frame(content, bg=self.colors["bg"])
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=2); grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1); grid.rowconfigure(1, weight=1)

        self.cam_card = self.create_card(grid, "FEED ANALYTICS", 0, 0)
        self.cam_label = tk.Label(self.cam_card, bg="black"); self.cam_label.pack(expand=True, padx=15, pady=10)

        self.graph_card = self.create_card(grid, "EMOTIONAL TRENDLINE", 1, 0)
        self.fig, self.ax = plt.subplots(figsize=(5, 3), dpi=90, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=10)
        
        self.reco_card = self.create_card(grid, "RECOMMEND ITEMS", 0, 1, rowspan=2)

    def create_stat(self, parent, title, val, col):
        f = tk.Frame(parent, bg=self.colors["card"], highlightbackground=self.colors["border"], highlightthickness=1)
        f.grid(row=0, column=col, padx=5, sticky="nsew"); parent.columnconfigure(col, weight=1)
        tk.Label(f, text=title, font=("Segoe UI", 8, "bold"), bg=self.colors["card"], fg=self.colors["subtext"]).pack(pady=(10, 0))
        l = tk.Label(f, text=val, font=("Segoe UI", 18, "bold"), bg=self.colors["card"], fg=self.colors["text"]); l.pack(pady=(0, 10)); return l

    def create_card(self, parent, title, r, c, rowspan=1):
        cont = tk.Frame(parent, bg=self.colors["bg"])
        cont.grid(row=r, column=c, rowspan=rowspan, sticky="nsew", padx=10, pady=10)
        tk.Label(cont, text=title, font=("Segoe UI", 9, "bold"), bg=self.colors["bg"], fg=self.colors["subtext"]).pack(anchor="w")
        card = tk.Frame(cont, bg=self.colors["card"], highlightbackground=self.colors["border"], highlightthickness=1)
        card.pack(fill="both", expand=True, pady=5); return card

    def build_prod_list(self, parent, prods):
        for p in prods:
            f = tk.Frame(parent, bg=self.colors["bg"], pady=8, padx=10, height=85)
            f.pack(fill="x", pady=4); f.pack_propagate(False)
            t = tk.Frame(f, bg=f['bg'], width=180)
            t.pack(side="left", fill="y"); t.pack_propagate(False)
            tk.Label(t, text=p['n'], font=("Segoe UI", 10, "bold"), bg=f['bg'], fg=self.colors["text"]).pack(anchor="w")
            tk.Label(t, text=p['p'], font=("Segoe UI", 9), bg=f['bg'], fg=self.colors["subtext"]).pack(anchor="w")
            img_c = tk.Frame(f, bg=f['bg'], width=65, height=65)
            img_c.pack(side="right"); img_c.pack_propagate(False)
            if p['i'] in self.image_cache:
                tk.Label(img_c, image=self.image_cache[p['i']], bg=f['bg']).pack()

    def create_recommendation_panels(self):
        self.calm_panel = tk.Frame(self.reco_card, bg=self.colors["card"])
        self.build_prod_list(self.calm_panel, PRODUCT_DATA["Calm"])
        self.stress_panel = tk.Frame(self.reco_card, bg=self.colors["card"])
        self.build_prod_list(self.stress_panel, PRODUCT_DATA["Stressed"])
        self.calm_panel.pack(fill="both", expand=True, padx=10, pady=10)

    def update_recommendations(self, emotion):
        if emotion == "Stressed":
            self.calm_panel.pack_forget()
            self.stress_panel.pack(fill="both", expand=True, padx=10, pady=10)
            self.reco_card.config(highlightbackground=self.CLR_STRESS, highlightthickness=2)
        else:
            self.stress_panel.pack_forget()
            self.calm_panel.pack(fill="both", expand=True, padx=10, pady=10)
            self.reco_card.config(highlightbackground=self.CLR_CALM, highlightthickness=2)

    def ai_worker(self, roi_gray):
        if model:
            roi = cv2.resize(roi_gray, (48, 48)) / 255.0
            roi = roi.reshape(1, 48, 48, 1)
            raw = model.predict(roi, verbose=0)[0]
            total = (raw[0]*1.0) + (raw[1]*1.2)
            self.result_queue.put([ (raw[0]*1.0)/total, (raw[1]*1.2)/total ])

    def update_app(self):
        if not self.running: return
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0 and self.frame_count % 5 == 0:
                x, y, w, h = faces[0]
                threading.Thread(target=self.ai_worker, args=(gray[y:y+h, x:x+w],), daemon=True).start()
            
            try: self.current_preds = self.result_queue.get_nowait()
            except queue.Empty: pass

            stress_val = int(self.current_preds[1]*100)
            raw_emo = "Stressed" if stress_val > 49 else "Calm"

            if raw_emo != self.stable_emotion:
                self.emotion_timer += 1
            else:
                self.emotion_timer = 0

            if self.emotion_timer >= self.STABILITY_THRESHOLD:
                self.stable_emotion = raw_emo
                self.update_recommendations(self.stable_emotion)
                self.emotion_timer = 0

            rect_color = (0, 0, 230) if self.stable_emotion == "Stressed" else (0, 200, 0)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
                cv2.putText(frame, self.stable_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_color, 2)

            self.stress_lbl.config(text=f"{stress_val}%", fg=self.CLR_STRESS if stress_val > 50 else self.CLR_CALM)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((400, 300), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.configure(image=imgtk); self.cam_label.image = imgtk

            if self.frame_count % 10 == 0:
                self.update_graph(self.current_preds)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = [current_time, self.stable_emotion, f"{stress_val}%"]
                for p in PRODUCT_DATA[self.stable_emotion]:
                    log_entry.append(p['n']); log_entry.append(p['p'])
                report_data.append(log_entry)

        self.window.after(15, self.update_app)

    def update_graph(self, preds):
        history["Calm"].append(preds[0]); history["Stressed"].append(preds[1])
        if len(history["Calm"]) > 20: 
            history["Calm"].pop(0); history["Stressed"].pop(0)
        self.ax.clear(); self.ax.set_facecolor(self.colors["card"])
        self.ax.plot(history["Calm"], color=self.CLR_CALM, lw=3, label="Calm")
        self.ax.plot(history["Stressed"], color=self.CLR_STRESS, lw=3, label="Stress")
        self.ax.set_ylim(-0.05, 1.05) 
        self.ax.legend(loc="upper right", fontsize='x-small', frameon=True)
        self.ax.grid(True, linestyle='--', alpha=0.3); self.canvas.draw()

    def view_summary(self):
        if not report_data:
            messagebox.showinfo("Summary", "No interaction data available yet.")
            return
            
        headers = ["Time", "Emotion", "Stress", "Item 1", "Price 1", "Item 2", "Price 2", "Item 3", "Price 3"]
        df = pd.DataFrame(report_data, columns=headers)
        
        # 1. Standardize the numeric data
        df['Stress_Val'] = df['Stress'].str.replace('%', '').astype(int)
        avg_stress = df['Stress_Val'].mean()
        
        # 2. Recalculate points based on the SAME 50% threshold
        calm_pts = len(df[df['Stress_Val'] <= 50])
        stress_pts = len(df[df['Stress_Val'] > 50])
        
        total_seconds_stressed = stress_pts * 0.2
        duration_text = f"Total Time Stressed: {total_seconds_stressed:.1f} Seconds"

        # 3. Unified Status Logic (Follows the Avg Stress number)
        if avg_stress > 50:
            insight = "STATUS: High Stress Detected"
            insight_clr = self.CLR_STRESS
        else:
            insight = "STATUS: Calm / Relaxed"
            insight_clr = self.CLR_CALM

        # --- UI and Chart Rendering ---
        sum_win = tk.Toplevel(self.window)
        sum_win.title("Session Analytics")
        sum_win.geometry("550x750")
        sum_win.configure(bg="white")

        fig_pie, ax_pie = plt.subplots(figsize=(5, 4), dpi=100)
        ax_pie.pie([calm_pts, stress_pts], labels=['Calm', 'Stressed'], 
                   colors=[self.CLR_CALM, self.CLR_STRESS], autopct='%1.1f%%', 
                   startangle=140, explode=(0.05, 0))
        ax_pie.set_title("Customer Mood Distribution", fontweight='bold', pad=20)
        
        canvas_pie = FigureCanvasTkAgg(fig_pie, master=sum_win)
        canvas_pie.get_tk_widget().pack(pady=10)
        canvas_pie.draw()

        tk.Label(sum_win, text=insight, bg="white", fg=insight_clr, font=("Segoe UI", 12, "bold")).pack(pady=5)
        tk.Label(sum_win, text=duration_text, bg="white", font=("Segoe UI", 11, "bold"), fg=self.CLR_STRESS).pack(pady=5)
        stats_txt = f"Total Samples: {len(df)}  |  Avg Stress: {avg_stress:.1f}%"
        tk.Label(sum_win, text=stats_txt, bg="white", font=("Segoe UI", 11)).pack(pady=5)

        def save_pie_image():
            timestamp = datetime.now().strftime('%H%M%S')
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png", 
                filetypes=[("PNG Image", "*.png")],
                initialfile=f"Retail_Mood_Summary_{timestamp}"
            )
            
            if file_path:
                summary_info = f"Samples: {len(df)} | {duration_text} | Avg Stress: {avg_stress:.1f}%"
                fig_pie.text(0.5, 0.02, summary_info, ha='center', fontsize=9, fontweight='bold', color="#374151")
                fig_pie.savefig(file_path, bbox_inches='tight', dpi=150)
                fig_pie.texts.clear()
                messagebox.showinfo("Export Success", "Diagram saved with analytics.")

        btn_frame = tk.Frame(sum_win, bg="white")
        btn_frame.pack(fill="x", padx=40, pady=20)

        tk.Button(btn_frame, text="SAVE DIAGRAM AS PNG", command=save_pie_image, 
                  bg=self.CLR_ACCENT, fg="white", font=("Segoe UI", 10, "bold"), 
                  pady=10, relief="flat", cursor="hand2").pack(fill="x", pady=5)
        
        tk.Button(btn_frame, text="CLOSE", command=sum_win.destroy, 
                  bg="#4B5563", fg="white", font=("Segoe UI", 10, "bold"), 
                  pady=10, relief="flat", cursor="hand2").pack(fill="x", pady=5)

    def reset_session(self):
        if messagebox.askyesno("Reset", "Clear all data for the current session?"):
            global report_data, history
            report_data.clear(); history = {"Calm": [], "Stressed": []}
            self.ax.clear(); self.canvas.draw()

    def download_report(self):
        if not report_data:
            messagebox.showwarning("No Data", "No session data recorded.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        default_name = f"Structured_Retail_Report_{timestamp}"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", 
            filetypes=[("CSV files", "*.csv")],
            initialfile=default_name
        )
        
        if file_path:
            try:
                headers = ["Timestamp", "Emotion State", "Stress Intensity", 
                           "Item 1", "Price 1", "Item 2", "Price 2", "Item 3", "Price 3"]
                df = pd.DataFrame(report_data, columns=headers)
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Report saved as:\n{os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")

    def on_closing(self):
        self.running = False
        if self.cap.isOpened(): self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk(); app = EmotionApp(root); root.mainloop()