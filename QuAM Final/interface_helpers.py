import tkinter as tk
from tkinter import ttk, messagebox

from configurations import (
    BG_COLOR,
    BUTTON_COLOR,
    BUTTON_HOVER_COLOR,
    BUTTON_PRESSED_COLOR,
    BUTTON_TEXT_COLOR,
    RISK_BUTTON_COLOR,
    RISK_BUTTON_HOVER_COLOR,
    RISK_BUTTON_PRESSED_COLOR,
    CLASSIFICATION_TASK,
    GAMMA_FIXED_OPTION,
    MAX_TREE_DEPTH,
    MAX_MIN_SAMPLES_LEAF,
)


def setup_button_styles(app):
    # Styling the main green buttons

    app.style = ttk.Style()
    app.style.theme_use("clam")
    app.style.configure(
        "QuAM.TButton",
        background=BUTTON_COLOR,
        foreground=BUTTON_TEXT_COLOR,
        borderwidth=0,
        focusthickness=0,
        focuscolor=BUTTON_COLOR,
        padding=(10, 6),
        font=("Arial", 13, "bold"),
    )
    app.style.map(
        "QuAM.TButton",
        background=[("active", BUTTON_HOVER_COLOR), ("pressed", BUTTON_PRESSED_COLOR)],
        foreground=[("active", BUTTON_TEXT_COLOR), ("pressed", BUTTON_TEXT_COLOR)],
    )

    # Styling the red risk button
    app.style.configure(
        "Risk.TButton",
        background=RISK_BUTTON_COLOR,
        foreground=BUTTON_TEXT_COLOR,
        borderwidth=0,
        focusthickness=0,
        focuscolor=RISK_BUTTON_COLOR,
        padding=(10, 6),
        font=("Arial", 13, "bold"),
    )
    app.style.map(
        "Risk.TButton",
        background=[("active", RISK_BUTTON_HOVER_COLOR), ("pressed", RISK_BUTTON_PRESSED_COLOR)],
        foreground=[("active", BUTTON_TEXT_COLOR), ("pressed", BUTTON_TEXT_COLOR)],
    )


def setup_scrollable_left_panel(parent):
    # Making left control panel scrollable

    left_container = tk.Frame(parent, bg=BG_COLOR)
    left_container.pack(side="left", fill="y", padx=(0, 10))

    left_canvas = tk.Canvas(
        left_container,
        bg=BG_COLOR,
        highlightthickness=0,
    )
    left_canvas.pack(side="left", fill="y")

    left_scrollbar = tk.Scrollbar(
        left_container,
        orient="vertical",
        command=left_canvas.yview,
    )
    left_scrollbar.pack(side="right", fill="y")

    left_canvas.configure(yscrollcommand=left_scrollbar.set)

    left_frame = tk.Frame(left_canvas, bg=BG_COLOR)
    left_window = left_canvas.create_window((0, 0), window=left_frame, anchor="nw")

    def update_scroll_region(event=None):
        left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        left_canvas.configure(width=left_frame.winfo_reqwidth())

    def resize_left_frame(event):
        left_canvas.itemconfig(left_window, width=event.width)

    # Mouse Scrolling Functions:
    def mousewheel_scroll(event):
        # Allows scrolling with mouse wheel / trackpad inside left panel

        if event.delta > 0:
            left_canvas.yview_scroll(-1, "units")
        elif event.delta < 0:
            left_canvas.yview_scroll(1, "units")

    def linux_scroll_up(event):
        left_canvas.yview_scroll(-1, "units")

    def linux_scroll_down(event):
        left_canvas.yview_scroll(1, "units")

    def bind_mousewheel(event):
        left_canvas.bind_all("<MouseWheel>", mousewheel_scroll)
        left_canvas.bind_all("<Button-4>", linux_scroll_up)
        left_canvas.bind_all("<Button-5>", linux_scroll_down)

    def unbind_mousewheel(event):
        left_canvas.unbind_all("<MouseWheel>")
        left_canvas.unbind_all("<Button-4>")
        left_canvas.unbind_all("<Button-5>")

    left_frame.bind("<Configure>", update_scroll_region)
    left_canvas.bind("<Configure>", resize_left_frame)
    left_canvas.bind("<Enter>", bind_mousewheel)
    left_canvas.bind("<Leave>", unbind_mousewheel)
    left_frame.bind("<Enter>", bind_mousewheel)
    left_frame.bind("<Leave>", unbind_mousewheel)

    return left_frame


def validate_model_params(params, task):
    # Checks parameter values before training

    try:
        random_state = int(params["random_state"])
    except Exception:
        messagebox.showerror("Invalid parameter", "Random seed must be an integer.")
        return False

    if random_state < 0:
        messagebox.showerror("Invalid parameter", "Random seed cannot be negative.")
        return False

    try:
        test_size = float(params["test_size"])
    except Exception:
        messagebox.showerror("Invalid parameter", "Evaluation split size must be a decimal number.")
        return False

    if test_size <= 0 or test_size >= 1:
        messagebox.showerror("Invalid parameter", "Evaluation split size must be between 0 and 1.")
        return False

    if task == CLASSIFICATION_TASK:
        try:
            c_value = float(params["c"])
        except Exception:
            messagebox.showerror("Invalid parameter", "Regularization parameter (C) must be a number.")
            return False

        if c_value <= 0:
            messagebox.showerror("Invalid parameter", "Regularization parameter (C) must be greater than 0.")
            return False

        if params["gamma_mode"] == GAMMA_FIXED_OPTION:
            try:
                gamma_value = float(params["gamma_fixed"])
            except Exception:
                messagebox.showerror("Invalid parameter", "Fixed gamma value must be a number.")
                return False

            if gamma_value <= 0:
                messagebox.showerror("Invalid parameter", "Fixed gamma value must be greater than 0.")
                return False

    else:
        try:
            max_depth = int(params["max_depth"])
        except Exception:
            messagebox.showerror("Invalid parameter", "Tree depth limit must be an integer.")
            return False

        if max_depth < 1:
            messagebox.showerror("Invalid parameter", "Tree depth limit must be at least 1.")
            return False

        if max_depth > MAX_TREE_DEPTH:
            messagebox.showerror("Invalid parameter", f"Tree depth limit cannot exceed {MAX_TREE_DEPTH}.")
            return False

        try:
            min_samples_leaf = int(params["min_samples_leaf"])
        except Exception:
            messagebox.showerror("Invalid parameter", "Minimum records per leaf must be an integer.")
            return False

        if min_samples_leaf < 1:
            messagebox.showerror("Invalid parameter", "Minimum records per leaf must be at least 1.")
            return False

        if min_samples_leaf > MAX_MIN_SAMPLES_LEAF:
            messagebox.showerror("Invalid parameter", f"Minimum records per leaf cannot exceed {MAX_MIN_SAMPLES_LEAF}.")
            return False

    return True