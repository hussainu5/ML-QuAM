import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import pandas as pd

from configurations import (
    BG_COLOR,
    PANEL_COLOR,
    BUTTON_TEXT_COLOR,
    ACCENT_COLOR,
    SUCCESS_COLOR,
    INFO_COLOR,
    MUTED_COLOR,
    OUTPUT_BG_COLOR,
    LIGHT_ENTRY_BG_COLOR,
    LIGHT_ENTRY_FG_COLOR,
    DARK_ENTRY_BG_COLOR,
    DARK_ENTRY_FG_COLOR,
    WINDOW_TITLE,
    MAIN_TITLE,
    SUBTITLE,
    CLASSIFICATION_TASK,
    REGRESSION_TASK,
    RANDOM_STATE_LABEL,
    TEST_SIZE_LABEL,
    C_LABEL,
    GAMMA_LABEL,
    FIXED_GAMMA_LABEL,
    CLASS_WEIGHT_LABEL,
    MAX_DEPTH_LABEL,
    MIN_SAMPLES_LEAF_LABEL,
    GAMMA_SCALE_OPTION,
    GAMMA_AUTO_OPTION,
    GAMMA_FIXED_OPTION,
    CLASS_WEIGHT_BALANCED_OPTION,
    CLASS_WEIGHT_NONE_OPTION,
)

from interface_helpers import (
    setup_button_styles,
    setup_scrollable_left_panel,
    validate_model_params,
)

from models import train_quam_model, predict_single_row, predict_all_delinquencies


class QuAMApp:
    def __init__(self, root):
        # Setting up the main app window

        self.root = root
        self.root.title(WINDOW_TITLE)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")

        self.root.configure(bg=BG_COLOR)

        setup_button_styles(self)

        # Main stored objects used across the app
        self.train_df = None
        self.query_df = None
        self.feature_df = None
        self.query_feature_df = None

        self.model = None
        self.pipeline = None

        self.current_task = None
        self.target_name = None
        self.last_metrics = None

        self._build_ui()

    def _build_ui(self):
        # Builds the overall layout

        title = tk.Label(
            self.root,
            text=MAIN_TITLE,
            font=("Arial", 21, "bold"),
            pady=10,
            bg=BG_COLOR,
            fg=ACCENT_COLOR,
        )
        title.pack()

        subtitle = tk.Label(
            self.root,
            text=SUBTITLE,
            font=("Arial", 15),
            pady=2,
            bg=BG_COLOR,
            fg=MUTED_COLOR,
        )
        subtitle.pack()

        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(fill="both", expand=True, padx=12, pady=10)

        left_frame = setup_scrollable_left_panel(main_frame)

        right_frame = tk.Frame(main_frame, bg=BG_COLOR)
        right_frame.pack(side="right", fill="both", expand=True)

        self._build_training_dataset_frame(left_frame)
        self._build_query_dataset_frame(left_frame)
        self._build_parameter_frame(left_frame)
        self._build_training_frame(left_frame)
        self._build_inference_frame(left_frame)
        self._build_output_frame(right_frame)

    def _build_training_dataset_frame(self, parent):
        # Labelled dataset section

        frame = tk.LabelFrame(
            parent,
            text="1. Labelled Training Dataset Input",
            padx=10,
            pady=10,
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        )
        frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            frame,
            text="Training dataset path:",
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        ).grid(row=0, column=0, sticky="w")

        self.train_path_var = tk.StringVar()

        train_path_entry = tk.Entry(
            frame,
            textvariable=self.train_path_var,
            width=45,
            bg=LIGHT_ENTRY_BG_COLOR,
            fg=LIGHT_ENTRY_FG_COLOR,
            insertbackground=LIGHT_ENTRY_FG_COLOR,
        )
        train_path_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=(4, 8))

        browse_btn = ttk.Button(
            frame,
            text="Browse",
            command=self.browse_training_file,
            width=12,
            style="QuAM.TButton",
        )
        browse_btn.grid(row=2, column=0, sticky="w")

        load_btn = ttk.Button(
            frame,
            text="Load Training Dataset",
            command=self.load_training_dataset,
            width=18,
            style="QuAM.TButton",
        )
        load_btn.grid(row=2, column=1, sticky="e")

        self.training_dataset_status_var = tk.StringVar(value="No training dataset loaded")

        status_lbl = tk.Label(
            frame,
            textvariable=self.training_dataset_status_var,
            fg=SUCCESS_COLOR,
            bg=PANEL_COLOR,
            wraplength=330,
            justify="left",
        )
        status_lbl.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_query_dataset_frame(self, parent):
        # Optional query dataset section

        frame = tk.LabelFrame(
            parent,
            text="2. Unlabelled Query Dataset Input (Optional)",
            padx=10,
            pady=10,
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        )
        frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            frame,
            text="Query dataset path:",
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        ).grid(row=0, column=0, sticky="w")

        self.query_path_var = tk.StringVar()

        query_path_entry = tk.Entry(
            frame,
            textvariable=self.query_path_var,
            width=45,
            bg=LIGHT_ENTRY_BG_COLOR,
            fg=LIGHT_ENTRY_FG_COLOR,
            insertbackground=LIGHT_ENTRY_FG_COLOR,
        )
        query_path_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=(4, 8))

        browse_btn = ttk.Button(
            frame,
            text="Browse",
            command=self.browse_query_file,
            width=12,
            style="QuAM.TButton",
        )
        browse_btn.grid(row=2, column=0, sticky="w")

        load_btn = ttk.Button(
            frame,
            text="Load Query Dataset",
            command=self.load_query_dataset,
            width=18,
            style="QuAM.TButton",
        )
        load_btn.grid(row=2, column=1, sticky="e")

        self.query_dataset_status_var = tk.StringVar(value="No query dataset loaded")

        status_lbl = tk.Label(
            frame,
            textvariable=self.query_dataset_status_var,
            fg=SUCCESS_COLOR,
            bg=PANEL_COLOR,
            wraplength=330,
            justify="left",
        )
        status_lbl.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_parameter_frame(self, parent):
        # Model parameter section

        frame = tk.LabelFrame(
            parent,
            text="3. Custom Parameter Configuration",
            padx=10,
            pady=10,
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        )
        frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            frame,
            text="Prediction task:",
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        ).grid(row=0, column=0, sticky="w")

        self.task_var = tk.StringVar(value=CLASSIFICATION_TASK)

        task_dropdown = tk.OptionMenu(
            frame,
            self.task_var,
            CLASSIFICATION_TASK,
            REGRESSION_TASK,
            command=lambda _: self.update_parameter_fields(),
        )
        task_dropdown.config(
            width=15,
            bg=LIGHT_ENTRY_BG_COLOR,
            fg=LIGHT_ENTRY_FG_COLOR,
            activebackground="#dfe7f1",
            activeforeground=LIGHT_ENTRY_FG_COLOR,
            highlightthickness=0,
            bd=1,
            font=("Arial", 15),
        )

        task_dropdown["menu"].config(
            bg=LIGHT_ENTRY_BG_COLOR,
            fg=LIGHT_ENTRY_FG_COLOR,
            activebackground="#dfe7f1",
            activeforeground=LIGHT_ENTRY_FG_COLOR,
            font=("Arial", 12),
        )
        task_dropdown.grid(row=0, column=1, sticky="w", pady=(0, 8))

        self.random_state_var = tk.StringVar(value="15288")
        self.test_size_var = tk.StringVar(value="0.2")

        tk.Label(
            frame,
            text=RANDOM_STATE_LABEL,
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        ).grid(row=1, column=0, sticky="w")

        tk.Entry(
            frame,
            textvariable=self.random_state_var,
            width=10,
            bg=DARK_ENTRY_BG_COLOR,
            fg=DARK_ENTRY_FG_COLOR,
            insertbackground=DARK_ENTRY_FG_COLOR,
        ).grid(row=1, column=1, sticky="w")

        tk.Label(
            frame,
            text=TEST_SIZE_LABEL,
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        ).grid(row=2, column=0, sticky="w")

        tk.Entry(
            frame,
            textvariable=self.test_size_var,
            width=10,
            bg=DARK_ENTRY_BG_COLOR,
            fg=DARK_ENTRY_FG_COLOR,
            insertbackground=DARK_ENTRY_FG_COLOR,
        ).grid(row=2, column=1, sticky="w", pady=(6, 8))

        self.dynamic_param_frame = tk.Frame(frame, bg=PANEL_COLOR)
        self.dynamic_param_frame.grid(row=3, column=0, columnspan=2, sticky="we")

        # Default parameter values shown in the input boxes
        self.c_var = tk.StringVar(value="1.0")
        self.gamma_mode_var = tk.StringVar(value=GAMMA_SCALE_OPTION)
        self.gamma_fixed_var = tk.StringVar(value="0.1")
        self.class_weight_var = tk.StringVar(value=CLASS_WEIGHT_BALANCED_OPTION)

        self.max_depth_var = tk.StringVar(value="5")
        self.min_samples_leaf_var = tk.StringVar(value="20")

        self.update_parameter_fields()

    def _build_training_frame(self, parent):
        # Training button section

        frame = tk.LabelFrame(
            parent,
            text="4. Model Training",
            padx=10,
            pady=10,
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        )
        frame.pack(fill="x", pady=(0, 10))

        train_btn = ttk.Button(
            frame,
            text="Train Model",
            command=self.train_model,
            width=18,
            style="QuAM.TButton",
        )
        train_btn.pack(anchor="w")

        self.training_status_var = tk.StringVar(value="Model not trained")

        tk.Label(
            frame,
            textvariable=self.training_status_var,
            fg=INFO_COLOR,
            bg=PANEL_COLOR,
            wraplength=330,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

    def _build_inference_frame(self, parent):
        # Inference section

        frame = tk.LabelFrame(
            parent,
            text="5. Inference",
            padx=10,
            pady=10,
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        )
        frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            frame,
            text="Customer row index:",
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        ).grid(row=0, column=0, sticky="w")

        self.row_index_var = tk.StringVar(value="0")

        tk.Entry(
            frame,
            textvariable=self.row_index_var,
            width=10,
            bg=DARK_ENTRY_BG_COLOR,
            fg=DARK_ENTRY_FG_COLOR,
            insertbackground=DARK_ENTRY_FG_COLOR,
        ).grid(row=0, column=1, sticky="w")

        predict_btn = ttk.Button(
            frame,
            text="Predict Selected Customer",
            command=self.predict_row,
            width=24,
            style="QuAM.TButton",
        )
        predict_btn.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.locate_delinquencies_btn = ttk.Button(
            frame,
            text="Locate All Predicted Delinquencies",
            command=self.locate_all_delinquencies,
            width=31,
            style="Risk.TButton",
        )
        self.locate_delinquencies_btn.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.update_delinquency_button_visibility()

    def _build_output_frame(self, parent):
        # Output log section

        frame = tk.LabelFrame(
            parent,
            text="Output Log",
            padx=10,
            pady=10,
            bg=PANEL_COLOR,
            fg=ACCENT_COLOR,
        )
        frame.pack(fill="both", expand=True)

        self.output_box = ScrolledText(
            frame,
            wrap="word",
            width=72,
            height=42,
            font=("Consolas", 14),
            bg=OUTPUT_BG_COLOR,
            fg=LIGHT_ENTRY_FG_COLOR,
        )
        self.output_box.pack(fill="both", expand=True)

        self.write_output(
            "Welcome to the Loan Repayment QuAM interface.\n\n"
            "Please refer to the user instruction manual (README) to learn how to use the interface.\n\n"
            "Below is a summary of the steps required to use the interface:\n"
            "1. Load a labelled training dataset.\n"
            "2. Optionally load a separate unlabelled query dataset. "
            "If a query dataset is loaded, inference will use that dataset. Otherwise it'll use the training dataset.\n"
            "3. Choose the prediction task and adjust the custom model parameters.\n"
            "4. Train the selected model.\n"
            "5. Enter a customer row index and generate an inference result.\n\n"
            "Classification target: default_flag (customer default / delinquency risk)\n"
            "Regression target: repayment_months (repayment latency)\n\n"
        )

    def browse_training_file(self):
        # Lets user choose training csv file

        filepath = filedialog.askopenfilename(
            title="Select Labelled Training Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if filepath:
            self.train_path_var.set(filepath)

    def browse_query_file(self):
        # Lets user choose query csv file

        filepath = filedialog.askopenfilename(
            title="Select Query Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if filepath:
            self.query_path_var.set(filepath)

    def load_training_dataset(self):
        # Loads labelled training data

        path = self.train_path_var.get().strip()

        if not path:
            messagebox.showwarning("Missing path", "Please enter or select a labelled training dataset path.")
            return

        try:
            self.train_df = pd.read_csv(path)

            self.training_dataset_status_var.set(
                f"Training dataset loaded successfully. Shape: {self.train_df.shape[0]} rows × {self.train_df.shape[1]} columns"
            )

            self.write_output(
                f"TRAINING DATASET LOADED\n"
                f"Path: {path}\n"
                f"Shape: {self.train_df.shape}\n"
                f"Columns preview: {list(self.train_df.columns[:15])}\n\n"
            )

        except Exception as e:
            messagebox.showerror("Load error", f"Could not load training dataset.\n\n{e}")

    def load_query_dataset(self):
        # Loads optional unlabelled query data
        path = self.query_path_var.get().strip()

        if not path:
            messagebox.showwarning("Missing path", "Please enter or select a query dataset path.")
            return

        try:
            self.query_df = pd.read_csv(path)

            self.query_dataset_status_var.set(
                f"Query dataset loaded successfully. Shape: {self.query_df.shape[0]} rows × {self.query_df.shape[1]} columns"
            )

            self.write_output(
                f"QUERY DATASET LOADED\n"
                f"Path: {path}\n"
                f"Shape: {self.query_df.shape}\n"
                f"Columns preview: {list(self.query_df.columns[:15])}\n\n"
            )

        except Exception as e:
            messagebox.showerror("Load error", f"Could not load query dataset.\n\n{e}")

    def update_parameter_fields(self, event=None):
        # Changing the parameters when user changes the task
        for widget in self.dynamic_param_frame.winfo_children():
            widget.destroy()

        task = self.task_var.get()

        if task == CLASSIFICATION_TASK:
            tk.Label(
                self.dynamic_param_frame,
                text="Model: Kernel SVM",
                bg=PANEL_COLOR,
                fg=ACCENT_COLOR,
            ).grid(row=0, column=0, columnspan=2, sticky="w")

            tk.Label(
                self.dynamic_param_frame,
                text=C_LABEL,
                bg=PANEL_COLOR,
                fg=ACCENT_COLOR,
            ).grid(row=1, column=0, sticky="w")

            tk.Entry(
                self.dynamic_param_frame,
                textvariable=self.c_var,
                width=12,
                bg=DARK_ENTRY_BG_COLOR,
                fg=DARK_ENTRY_FG_COLOR,
                insertbackground=DARK_ENTRY_FG_COLOR,
            ).grid(row=1, column=1, sticky="w")

            tk.Label(
                self.dynamic_param_frame,
                text=GAMMA_LABEL,
                bg=PANEL_COLOR,
                fg=ACCENT_COLOR,
            ).grid(row=2, column=0, sticky="w")

            gamma_dropdown = tk.OptionMenu(
                self.dynamic_param_frame,
                self.gamma_mode_var,
                GAMMA_SCALE_OPTION,
                GAMMA_AUTO_OPTION,
                GAMMA_FIXED_OPTION,
                command=lambda _: self.update_gamma_fixed_field(),
            )
            gamma_dropdown.config(
                width=10,
                bg=LIGHT_ENTRY_BG_COLOR,
                fg=LIGHT_ENTRY_FG_COLOR,
                activebackground="#dfe7f1",
                activeforeground=LIGHT_ENTRY_FG_COLOR,
                highlightthickness=0,
                bd=1,
                font=("Arial", 12),
            )
            gamma_dropdown["menu"].config(
                bg=LIGHT_ENTRY_BG_COLOR,
                fg=LIGHT_ENTRY_FG_COLOR,
                activebackground="#dfe7f1",
                activeforeground=LIGHT_ENTRY_FG_COLOR,
                font=("Arial", 12),
            )
            gamma_dropdown.grid(row=2, column=1, sticky="w")

            self.fixed_gamma_label = tk.Label(
                self.dynamic_param_frame,
                text=FIXED_GAMMA_LABEL,
                bg=PANEL_COLOR,
                fg=ACCENT_COLOR,
            )

            self.fixed_gamma_entry = tk.Entry(
                self.dynamic_param_frame,
                textvariable=self.gamma_fixed_var,
                width=12,
                bg=DARK_ENTRY_BG_COLOR,
                fg=DARK_ENTRY_FG_COLOR,
                insertbackground=DARK_ENTRY_FG_COLOR,
            )

            self.class_weight_label = tk.Label(
                self.dynamic_param_frame,
                text=CLASS_WEIGHT_LABEL,
                bg=PANEL_COLOR,
                fg=ACCENT_COLOR,
            )

            self.class_weight_dropdown = tk.OptionMenu(
                self.dynamic_param_frame,
                self.class_weight_var,
                CLASS_WEIGHT_BALANCED_OPTION,
                CLASS_WEIGHT_NONE_OPTION,
            )
            self.class_weight_dropdown.config(
                width=22,
                bg=LIGHT_ENTRY_BG_COLOR,
                fg=LIGHT_ENTRY_FG_COLOR,
                activebackground="#dfe7f1",
                activeforeground=LIGHT_ENTRY_FG_COLOR,
                highlightthickness=0,
                bd=1,
                font=("Arial", 12),
            )
            self.class_weight_dropdown["menu"].config(
                bg=LIGHT_ENTRY_BG_COLOR,
                fg=LIGHT_ENTRY_FG_COLOR,
                activebackground="#dfe7f1",
                activeforeground=LIGHT_ENTRY_FG_COLOR,
                font=("Arial", 12),
            )

            self.update_gamma_fixed_field()

        else:
            tk.Label(
                self.dynamic_param_frame,
                text="Model: Decision Tree Regressor",
                bg=PANEL_COLOR,
                fg=ACCENT_COLOR,
            ).grid(row=0, column=0, columnspan=2, sticky="w")

            tk.Label(
                self.dynamic_param_frame,
                text=MAX_DEPTH_LABEL,
                bg=PANEL_COLOR,
                fg=ACCENT_COLOR,
            ).grid(row=1, column=0, sticky="w")

            tk.Entry(
                self.dynamic_param_frame,
                textvariable=self.max_depth_var,
                width=12,
                bg=DARK_ENTRY_BG_COLOR,
                fg=DARK_ENTRY_FG_COLOR,
                insertbackground=DARK_ENTRY_FG_COLOR,
            ).grid(row=1, column=1, sticky="w")

            tk.Label(
                self.dynamic_param_frame,
                text=MIN_SAMPLES_LEAF_LABEL,
                bg=PANEL_COLOR,
                fg=ACCENT_COLOR,
            ).grid(row=2, column=0, sticky="w")

            tk.Entry(
                self.dynamic_param_frame,
                textvariable=self.min_samples_leaf_var,
                width=12,
                bg=DARK_ENTRY_BG_COLOR,
                fg=DARK_ENTRY_FG_COLOR,
                insertbackground=DARK_ENTRY_FG_COLOR,
            ).grid(row=2, column=1, sticky="w")

        self.update_delinquency_button_visibility()

    def update_gamma_fixed_field(self):
        # Shows fixed gamma input only when fixed is selected

        if not hasattr(self, "fixed_gamma_label"):
            return

        if self.gamma_mode_var.get() == GAMMA_FIXED_OPTION:
            self.fixed_gamma_label.grid(row=3, column=0, sticky="w")
            self.fixed_gamma_entry.grid(row=3, column=1, sticky="w")

            self.class_weight_label.grid(row=4, column=0, sticky="w")
            self.class_weight_dropdown.grid(row=4, column=1, sticky="w")

        else:
            self.fixed_gamma_label.grid_remove()
            self.fixed_gamma_entry.grid_remove()

            self.class_weight_label.grid(row=3, column=0, sticky="w")
            self.class_weight_dropdown.grid(row=3, column=1, sticky="w")

    def update_delinquency_button_visibility(self):
        # Shows delinquency locator only for classification

        if hasattr(self, "locate_delinquencies_btn"):
            if self.task_var.get() == CLASSIFICATION_TASK:
                self.locate_delinquencies_btn.grid()
            else:
                self.locate_delinquencies_btn.grid_remove()

    def collect_model_params(self):
        # Collects parameter values before training
        params = {
            "random_state": self.random_state_var.get(),
            "test_size": self.test_size_var.get(),
            "c": self.c_var.get(),
            "gamma_mode": self.gamma_mode_var.get(),
            "gamma_fixed": self.gamma_fixed_var.get(),
            "class_weight": self.class_weight_var.get(),
            "max_depth": self.max_depth_var.get(),
            "min_samples_leaf": self.min_samples_leaf_var.get(),
        }

        return params

    def train_model(self):
        # Runs model training from models.py

        if self.train_df is None:
            messagebox.showwarning("No dataset", "Please load a labelled training dataset first.")
            return

        task = self.task_var.get()
        params = self.collect_model_params()

        if not validate_model_params(params, task):
            return

        try:
            result = train_quam_model(
                train_df=self.train_df,
                query_df=self.query_df,
                task=task,
                params=params,
            )

            self.current_task = task
            self.model = result["model"]
            self.pipeline = result["pipeline"]
            self.feature_df = result["feature_df"]
            self.query_feature_df = result["query_feature_df"]
            self.target_name = result["target_name"]
            self.last_metrics = result["metrics"]

            self.write_output(result["output_text"])

            if task == CLASSIFICATION_TASK:
                self.training_status_var.set("Classification model trained successfully")
            else:
                self.training_status_var.set("Regression model trained successfully")

        except Exception as e:
            messagebox.showerror("Training error", f"Could not train the model.\n\n{e}")

    def predict_row(self):
        # Predicts selected row from query data if loaded, otherwise training data

        if self.pipeline is None or self.feature_df is None:
            messagebox.showwarning("Model not ready", "Please load data and train a model first.")
            return

        try:
            row_index = int(self.row_index_var.get())

        except Exception:
            messagebox.showwarning("Invalid index", "Please enter a valid integer row index.")
            return

        if self.query_feature_df is not None:
            active_df = self.query_feature_df
            dataset_name = "query dataset"
        else:
            active_df = self.feature_df
            dataset_name = "training dataset"

        if row_index not in active_df.index:
            messagebox.showwarning("Row not found", f"The row index is not present in the active {dataset_name}.")
            return

        try:
            row_df = active_df.loc[[row_index]].copy()
            result = predict_single_row(self.pipeline, row_df, self.current_task)

            if self.current_task == CLASSIFICATION_TASK:
                self.write_output(
                    "INFERENCE RESULT\n"
                    f"Task: {self.current_task}\n"
                    f"Source: {dataset_name}\n"
                    f"Customer row index: {row_index}\n"
                    f"Predicted class: {result['pred_class']} ({result['label']})\n"
                    f"Predicted delinquency probability: {result['pred_prob']:.4f}\n\n"
                )

            else:
                self.write_output(
                    "INFERENCE RESULT\n"
                    f"Task: {self.current_task}\n"
                    f"Source: {dataset_name}\n"
                    f"Customer row index: {row_index}\n"
                    f"Predicted repayment latency: {result['pred_value']:.2f} months\n\n"
                )

        except Exception as e:
            messagebox.showerror("Inference error", f"Could not generate prediction.\n\n{e}")

    def locate_all_delinquencies(self):
        # Finds all customers predicted as delinquent

        if self.pipeline is None or self.feature_df is None:
            messagebox.showwarning("Model not ready", "Please load data and train a classification model first.")
            return

        if self.current_task != CLASSIFICATION_TASK:
            messagebox.showwarning("Wrong task", "Please train a classification model before locating delinquencies.")
            return

        if self.query_feature_df is not None:
            active_df = self.query_feature_df
            dataset_name = "query dataset"
        else:
            active_df = self.feature_df
            dataset_name = "training dataset"

        try:
            delinquency_results = predict_all_delinquencies(self.pipeline, active_df)

            self.write_output(
                "PREDICTED DELINQUENCY LIST\n"
                f"Task: {self.current_task}\n"
                f"Source: {dataset_name}\n"
                f"Rows scanned: {len(active_df)}\n"
                f"Predicted delinquencies found: {len(delinquency_results)}\n\n"
            )

            if len(delinquency_results) == 0:
                self.write_output("No customers were predicted as delinquent.\n\n")
                return

            for result in delinquency_results:
                self.write_output(
                    f"Customer row index: {result['row_index']} | "
                    f"Predicted delinquency probability: {result['pred_prob']:.4f}\n"
                )

            self.write_output("\n")

        except Exception as e:
            messagebox.showerror("Inference error", f"Could not locate predicted delinquencies.\n\n{e}")

    def write_output(self, text):
        self.output_box.insert("end", text)
        self.output_box.see("end")