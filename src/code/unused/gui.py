import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#Maybe make a GUI
class AnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis Interface")
        self.root.geometry("1200x800")

        # Initialize data storage
        self.data = None
        self.current_figure = None

        # Create main layout
        self.create_layout()
        self.create_widgets()

    def create_layout(self):
        # Create main containers
        self.left_panel = ttk.Frame(self.root, padding="10")
        self.left_panel.grid(row=0, column=0, sticky="nsew")

        self.right_panel = ttk.Frame(self.root, padding="10")
        self.right_panel.grid(row=0, column=1, sticky="nsew")

        # Configure grid weights
        self.root.grid_columnconfigure(1, weight=1)  # Right panel expands
        self.root.grid_rowconfigure(0, weight=1)  # Both panels expand vertically

    def create_widgets(self):
        # Data Loading Section
        data_frame = ttk.LabelFrame(self.left_panel, text="Data Management", padding="5")
        data_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(data_frame, text="Load New Data",
                   command=self.load_new_data).pack(fill="x", padx=5, pady=2)
        ttk.Button(data_frame, text="Load Prepared Data",
                   command=self.load_prepared_data).pack(fill="x", padx=5, pady=2)

        # Visualization Section
        viz_frame = ttk.LabelFrame(self.left_panel, text="Visualization", padding="5")
        viz_frame.pack(fill="x", padx=5, pady=5)

        self.plot_type = ttk.Combobox(viz_frame,
                                      values=['Histogram', 'Box Plot', 'Scatter Plot', 'Line Plot'])
        self.plot_type.pack(fill="x", padx=5, pady=2)
        self.plot_type.set('Select Plot Type')

        ttk.Button(viz_frame, text="Generate Plot",
                   command=self.generate_plot).pack(fill="x", padx=5, pady=2)

        # Analysis Section
        analysis_frame = ttk.LabelFrame(self.left_panel, text="Analysis", padding="5")
        analysis_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(analysis_frame, text="Run NLP Analysis",
                   command=self.run_nlp_analysis).pack(fill="x", padx=5, pady=2)
        ttk.Button(analysis_frame, text="Run Sentiment Analysis",
                   command=self.run_sentiment_analysis).pack(fill="x", padx=5, pady=2)

        # Results area in right panel
        self.results_frame = ttk.Frame(self.right_panel)
        self.results_frame.pack(fill="both", expand=True)

    def load_new_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[('Text Files', '*.txt'), ('CSV Files', '*.csv'), ('All Files', '*.*')])
        if file_path:
            try:
                # Here you can add your own data loading logic
                if file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                    # Placeholder for your preprocessing
                    self.data = pd.DataFrame({'text': [text]})
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def load_prepared_data(self):
        file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Prepared data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def clear_display(self):
        # Clear the current display
        if self.current_figure:
            plt.close(self.current_figure)
        for widget in self.results_frame.winfo_children():
            widget.destroy()

    def generate_plot(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        self.clear_display()

        try:
            # Create new figure
            self.current_figure, ax = plt.subplots(figsize=(10, 6))
            plot_type = self.plot_type.get()

            if plot_type == 'Histogram':
                # Example plotting - replace with your visualization logic
                sns.histplot(data=self.data.select_dtypes(include=['number']).iloc[:, 0], ax=ax)
            elif plot_type == 'Box Plot':
                sns.boxplot(data=self.data.select_dtypes(include=['number']), ax=ax)
            elif plot_type == 'Scatter Plot':
                num_cols = self.data.select_dtypes(include=['number']).columns[:2]
                if len(num_cols) >= 2:
                    sns.scatterplot(data=self.data, x=num_cols[0], y=num_cols[1], ax=ax)
            elif plot_type == 'Line Plot':
                sns.lineplot(data=self.data.select_dtypes(include=['number']).iloc[:, 0], ax=ax)

            # Display the plot
            canvas = FigureCanvasTkAgg(self.current_figure, master=self.results_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Error generating plot: {str(e)}")

    def run_nlp_analysis(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        # Placeholder for your NLP analysis implementation
        messagebox.showinfo("Info", "Replace this with your NLP analysis implementation")

    def run_sentiment_analysis(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        # Placeholder for your sentiment analysis implementation
        messagebox.showinfo("Info", "Replace this with your sentiment analysis implementation")


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = AnalysisGUI(root)
#     root.mainloop()