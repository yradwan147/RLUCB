#!/usr/bin/env python3
"""
UCB Quiz Master - GUI Application with real-time UCB algorithm visualization.
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
import threading
from quiz_manager import QuizManager
import config


class UCBVisualizationPanel:
    """Panel for visualizing the UCB algorithm in real-time."""
    
    def __init__(self, parent_frame):
        """Initialize the visualization panel."""
        self.frame = parent_frame
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plot
        self.ax.set_title("UCB Algorithm Visualization", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Subtopics", fontsize=11)
        self.ax.set_ylabel("Score", fontsize=11)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_ylim(0, 2)
        
        self.canvas.draw()
    
    def update_plot(self, quiz_manager):
        """
        Update the UCB visualization with current data.
        
        Args:
            quiz_manager: The QuizManager instance with current state
        """
        if not quiz_manager.ucb_selector:
            return
        
        # Get UCB data
        ucb_data = quiz_manager.ucb_selector.get_ucb_data_for_visualization()
        subtopics = quiz_manager.subtopics
        
        # Clear previous plot
        self.ax.clear()
        
        # Prepare data for plotting
        x_positions = list(range(len(subtopics)))
        weakness_scores = []
        ucb_scores = []
        correctness_rates = []
        attempts = []
        
        for subtopic in subtopics:
            data = ucb_data[subtopic]
            weakness_scores.append(data['weakness_score'])
            
            # Cap UCB score at 2 for visualization
            ucb_score = min(data['ucb_score'], 2.0) if data['ucb_score'] != float('inf') else 2.0
            ucb_scores.append(ucb_score)
            
            correctness_rates.append(data['correctness_rate'])
            attempts.append(data['attempts'])
        
        # Create the box plot visualization
        colors = []
        for i, (weakness, ucb, attempts_count) in enumerate(zip(weakness_scores, ucb_scores, attempts)):
            # Color based on performance
            if attempts_count == 0:
                color = 'gray'
            elif correctness_rates[i] < 0.5:
                color = 'red'
            elif correctness_rates[i] < 0.75:
                color = 'orange'
            else:
                color = 'green'
            colors.append(color)
            
            # Draw the "box" - a vertical line from weakness score to UCB score
            self.ax.plot([i, i], [weakness, ucb], color=color, linewidth=3, alpha=0.6)
            
            # Draw the center point (empirical weakness score)
            self.ax.scatter([i], [weakness], color=color, s=150, zorder=5, 
                          edgecolors='black', linewidths=2, marker='o')
            
            # Draw the upper whisker (UCB bound with exploration)
            self.ax.scatter([i], [ucb], color=color, s=100, zorder=5, 
                          marker='_', linewidths=3)
            
            # Add exploration bonus indicator
            exploration_bonus = ucb - weakness
            if exploration_bonus > 0.05:  # Only show if significant
                self.ax.annotate(f'+{exploration_bonus:.2f}', 
                               xy=(i, ucb), 
                               xytext=(i, ucb + 0.1),
                               ha='center', 
                               fontsize=8,
                               color=color,
                               fontweight='bold')
        
        # Highlight the next selected subtopic
        next_subtopic = quiz_manager.ucb_selector.select_next_subtopic()
        next_idx = subtopics.index(next_subtopic)
        self.ax.axvspan(next_idx - 0.3, next_idx + 0.3, 
                       alpha=0.2, color='yellow', zorder=0)
        
        # Set labels and formatting
        self.ax.set_xticks(x_positions)
        
        # Truncate long subtopic names for display
        labels = [subtopic[:20] + '...' if len(subtopic) > 20 else subtopic 
                 for subtopic in subtopics]
        self.ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        
        self.ax.set_ylabel("Score (0 = Perfect, 1 = No Knowledge, >1 = Needs Exploration)", fontsize=9)
        self.ax.set_xlabel("Subtopics", fontsize=11, fontweight='bold')
        
        title = "UCB Algorithm: Weakness Score (●) + Exploration Bonus (|)"
        self.ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        self.ax.set_ylim(-0.1, 2.1)
        self.ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Good (>75%)', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=10, label='Moderate (50-75%)', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Weak (<50%)', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=10, label='Not Attempted', markeredgecolor='black'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Redraw canvas
        self.canvas.draw()


class QuizGUI:
    """Main GUI application for the UCB Quiz Master."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("🎓 UCB Quiz Master - Adaptive Learning System")
        self.root.geometry("1400x900")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Application state
        self.quiz = None
        self.current_question = None
        self.debug_mode = tk.BooleanVar(value=False)
        self.question_count = 0
        self.selected_answer = tk.StringVar()
        
        # Build UI
        self.create_widgets()
        
        # Show setup screen
        self.show_setup_screen()
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # Top bar with debug mode toggle
        top_bar = ttk.Frame(main_container)
        top_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.title_label = ttk.Label(top_bar, text="🎓 UCB Quiz Master", 
                                     font=('Arial', 20, 'bold'))
        self.title_label.pack(side=tk.LEFT)
        
        debug_check = ttk.Checkbutton(top_bar, text="Debug Mode (Show UCB Visualization)", 
                                     variable=self.debug_mode, 
                                     command=self.toggle_debug_mode)
        debug_check.pack(side=tk.RIGHT, padx=10)
        
        # Main content area with paned window
        self.paned_window = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        self.paned_window.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left side: Quiz interface
        self.quiz_frame = ttk.Frame(self.paned_window, padding="10")
        self.paned_window.add(self.quiz_frame, weight=2)
        
        # Right side: Debug visualization (hidden by default)
        self.debug_frame = ttk.Frame(self.paned_window, padding="10")
        self.viz_panel = UCBVisualizationPanel(self.debug_frame)
        
        # Setup screen (will be replaced with quiz screen)
        self.setup_frame = None
        self.question_frame = None
    
    def show_setup_screen(self):
        """Display the initial setup screen."""
        if self.setup_frame:
            self.setup_frame.destroy()
        
        self.setup_frame = ttk.Frame(self.quiz_frame)
        self.setup_frame.pack(fill=tk.BOTH, expand=True)
        
        # Welcome message
        welcome = ttk.Label(self.setup_frame, 
                          text="Welcome to UCB Quiz Master!\n\nAdaptive Learning with Upper Confidence Bound Algorithm", 
                          font=('Arial', 16, 'bold'),
                          justify=tk.CENTER)
        welcome.pack(pady=30)
        
        # API key check
        if not config.OPENAI_API_KEY:
            error_label = ttk.Label(self.setup_frame, 
                                   text="❌ Error: OpenAI API key not configured!\n\n"
                                        "Please create a .env file with:\nOPENAI_API_KEY=your_key_here",
                                   font=('Arial', 12),
                                   foreground='red',
                                   justify=tk.CENTER)
            error_label.pack(pady=20)
            return
        
        # Topic input
        topic_frame = ttk.Frame(self.setup_frame)
        topic_frame.pack(pady=20)
        
        ttk.Label(topic_frame, text="Enter Topic:", font=('Arial', 12)).pack()
        self.topic_entry = ttk.Entry(topic_frame, width=40, font=('Arial', 12))
        self.topic_entry.pack(pady=10)
        self.topic_entry.focus()
        
        # Subtopics count
        subtopic_frame = ttk.Frame(self.setup_frame)
        subtopic_frame.pack(pady=10)
        
        ttk.Label(subtopic_frame, text="Number of Subtopics:", font=('Arial', 11)).pack()
        self.subtopic_spinbox = ttk.Spinbox(subtopic_frame, from_=2, to=10, 
                                           width=10, font=('Arial', 11))
        self.subtopic_spinbox.set(5)
        self.subtopic_spinbox.pack(pady=5)
        
        # Start button
        start_btn = ttk.Button(self.setup_frame, text="🚀 Start Quiz", 
                              command=self.start_quiz,
                              style='Accent.TButton')
        start_btn.pack(pady=30)
        
        # Bind Enter key
        self.topic_entry.bind('<Return>', lambda e: self.start_quiz())
        
        # Info text
        info_text = ("This application will:\n"
                    "• Generate subtopics for your topic\n"
                    "• Create 10 questions per subtopic\n"
                    "• Adapt to find your weak areas\n"
                    "• Focus on helping you improve\n\n"
                    "Enable Debug Mode to see the UCB algorithm in action!")
        
        info_label = ttk.Label(self.setup_frame, text=info_text, 
                             font=('Arial', 10),
                             justify=tk.LEFT,
                             foreground='gray')
        info_label.pack(pady=20)
    
    def start_quiz(self):
        """Initialize and start the quiz."""
        topic = self.topic_entry.get().strip()
        if not topic:
            messagebox.showerror("Error", "Please enter a topic!")
            return
        
        try:
            num_subtopics = int(self.subtopic_spinbox.get())
        except ValueError:
            num_subtopics = 5
        
        # Show loading message
        self.setup_frame.destroy()
        loading_label = ttk.Label(self.quiz_frame, 
                                 text=f"🤖 Initializing quiz on '{topic}'...\n\n"
                                      "Generating subtopics and questions...\n"
                                      "This may take a minute.",
                                 font=('Arial', 14),
                                 justify=tk.CENTER)
        loading_label.pack(expand=True)
        
        # Initialize quiz in background thread
        def init_quiz():
            try:
                self.quiz = QuizManager(topic)
                self.quiz.initialize(num_subtopics)
                
                # Update UI in main thread
                self.root.after(0, self.show_quiz_screen)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", 
                    f"Failed to initialize quiz:\n{str(e)}\n\n"
                    "Please check your API key and internet connection."
                ))
                self.root.after(0, self.show_setup_screen)
        
        thread = threading.Thread(target=init_quiz, daemon=True)
        thread.start()
    
    def show_quiz_screen(self):
        """Display the main quiz interface."""
        # Clear quiz frame
        for widget in self.quiz_frame.winfo_children():
            widget.destroy()
        
        # Create question display area
        self.question_frame = ttk.Frame(self.quiz_frame)
        self.question_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Question header
        header_frame = ttk.Frame(self.question_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.question_num_label = ttk.Label(header_frame, text="Question 1", 
                                           font=('Arial', 16, 'bold'))
        self.question_num_label.pack(side=tk.LEFT)
        
        self.subtopic_label = ttk.Label(header_frame, text="", 
                                       font=('Arial', 12),
                                       foreground='blue')
        self.subtopic_label.pack(side=tk.RIGHT)
        
        # Question text
        self.question_text = scrolledtext.ScrolledText(self.question_frame, 
                                                      wrap=tk.WORD, 
                                                      height=5,
                                                      font=('Arial', 13),
                                                      relief=tk.SOLID,
                                                      borderwidth=1)
        self.question_text.pack(fill=tk.X, pady=(0, 20))
        
        # Options frame
        self.options_frame = ttk.LabelFrame(self.question_frame, text="Select your answer:", 
                                           padding="15")
        self.options_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Create radio buttons for options
        self.option_buttons = []
        for option in ['A', 'B', 'C', 'D']:
            rb = ttk.Radiobutton(self.options_frame, text="", 
                               variable=self.selected_answer, 
                               value=option,
                               style='Large.TRadiobutton')
            rb.pack(anchor=tk.W, pady=8, padx=10, fill=tk.X)
            self.option_buttons.append(rb)
        
        # Result display area (hidden initially)
        self.result_frame = ttk.Frame(self.question_frame)
        
        # Control buttons
        button_frame = ttk.Frame(self.question_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.submit_btn = ttk.Button(button_frame, text="Submit Answer", 
                                    command=self.submit_answer)
        self.submit_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = ttk.Button(button_frame, text="Next Question →", 
                                   command=self.next_question)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn.config(state='disabled')
        
        stats_btn = ttk.Button(button_frame, text="📊 View Statistics", 
                              command=self.show_statistics)
        stats_btn.pack(side=tk.RIGHT, padx=5)
        
        # Load first question
        self.next_question()
        
        # Update visualization if debug mode is on
        if self.debug_mode.get():
            self.viz_panel.update_plot(self.quiz)
    
    def next_question(self):
        """Load and display the next question."""
        self.question_count += 1
        
        # Get next question
        self.current_question = self.quiz.get_next_question()
        
        if not self.current_question:
            self.show_completion_screen()
            return
        
        # Update UI
        self.question_num_label.config(text=f"Question {self.question_count}")
        self.subtopic_label.config(text=f"📚 {self.quiz.current_subtopic}")
        
        # Display question
        self.question_text.config(state='normal')
        self.question_text.delete(1.0, tk.END)
        self.question_text.insert(1.0, self.current_question['question'])
        self.question_text.config(state='disabled')
        
        # Display options
        for i, (option, text) in enumerate(sorted(self.current_question['options'].items())):
            self.option_buttons[i].config(text=f"{option}. {text}")
        
        # Reset state
        self.selected_answer.set('')
        self.submit_btn.config(state='normal')
        self.next_btn.config(state='disabled')
        
        # Hide result frame
        self.result_frame.pack_forget()
        
        # Update visualization if debug mode is on
        if self.debug_mode.get():
            self.viz_panel.update_plot(self.quiz)
    
    def submit_answer(self):
        """Submit the current answer."""
        answer = self.selected_answer.get()
        
        if not answer:
            messagebox.showwarning("Warning", "Please select an answer!")
            return
        
        # Submit answer to quiz manager
        result = self.quiz.submit_answer(answer)
        
        # Display result
        self.show_result(result)
        
        # Update button states
        self.submit_btn.config(state='disabled')
        self.next_btn.config(state='normal')
        
        # Update visualization if debug mode is on
        if self.debug_mode.get():
            self.viz_panel.update_plot(self.quiz)
    
    def show_result(self, result):
        """Display the result of the submitted answer."""
        # Clear previous result
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        if result['is_correct']:
            icon = "✅"
            message = "Correct! Well done!"
            color = 'green'
        else:
            icon = "❌"
            message = f"Incorrect. The correct answer is: {result['correct_answer']}"
            color = 'red'
        
        result_label = ttk.Label(self.result_frame, text=f"{icon} {message}", 
                                font=('Arial', 12, 'bold'),
                                foreground=color)
        result_label.pack(pady=(0, 10))
        
        # Show explanation for incorrect answers
        if not result['is_correct']:
            exp_frame = ttk.LabelFrame(self.result_frame, text="💡 Explanation:", 
                                      padding="10")
            exp_frame.pack(fill=tk.BOTH, expand=True)
            
            exp_text = scrolledtext.ScrolledText(exp_frame, wrap=tk.WORD, 
                                                height=4,
                                                font=('Arial', 11),
                                                relief=tk.FLAT,
                                                background='#f0f0f0')
            exp_text.pack(fill=tk.BOTH, expand=True)
            exp_text.insert(1.0, result['explanation'])
            exp_text.config(state='disabled')
    
    def show_statistics(self):
        """Show detailed statistics in a popup window."""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("📊 Learning Progress Statistics")
        stats_window.geometry("600x500")
        
        # Get statistics
        summary = self.quiz.get_progress_summary()
        
        # Display in text widget
        text_widget = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD, 
                                               font=('Courier', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        text_widget.insert(1.0, summary)
        text_widget.config(state='disabled')
        
        # Close button
        close_btn = ttk.Button(stats_window, text="Close", 
                              command=stats_window.destroy)
        close_btn.pack(pady=10)
    
    def show_completion_screen(self):
        """Show the quiz completion screen."""
        # Clear question frame
        for widget in self.question_frame.winfo_children():
            widget.destroy()
        
        completion_frame = ttk.Frame(self.question_frame)
        completion_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(completion_frame, text="🎉 Quiz Complete!", 
                 font=('Arial', 24, 'bold'),
                 foreground='green').pack(pady=30)
        
        stats_text = self.quiz.get_progress_summary()
        
        text_widget = scrolledtext.ScrolledText(completion_frame, wrap=tk.WORD, 
                                               font=('Courier', 10),
                                               height=15)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        text_widget.insert(1.0, stats_text)
        text_widget.config(state='disabled')
        
        # Restart button
        restart_btn = ttk.Button(completion_frame, text="Start New Quiz", 
                                command=self.show_setup_screen)
        restart_btn.pack(pady=20)
    
    def toggle_debug_mode(self):
        """Toggle debug mode visualization."""
        if self.debug_mode.get():
            # Show debug panel
            self.paned_window.add(self.debug_frame, weight=1)
            
            # Update visualization if quiz is active
            if self.quiz and self.quiz.ucb_selector:
                self.viz_panel.update_plot(self.quiz)
        else:
            # Hide debug panel
            self.paned_window.remove(self.debug_frame)


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    
    # Check if API key is configured
    if not config.OPENAI_API_KEY:
        messagebox.showerror(
            "Configuration Error",
            "OpenAI API key not configured!\n\n"
            "Please create a .env file with:\n"
            "OPENAI_API_KEY=your_key_here"
        )
    
    app = QuizGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

