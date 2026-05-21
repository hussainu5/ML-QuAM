# Interface Configs:
# Titles
WINDOW_TITLE = "Loan Repayment QuAM"
MAIN_TITLE = "Loan Repayment QuAM: Predicting Customer Defaults (Classification) & Repayment Latency (Regression)"
SUBTITLE = "Datasets Input | Custom Parameter Configuration | Model Training | Inference"

# Colors
BG_COLOR = "#f4f7fb"
PANEL_COLOR = "#ffffff"

BUTTON_COLOR = "#21c44a"
BUTTON_HOVER_COLOR = "#18a73d"
BUTTON_PRESSED_COLOR = "#159437"
BUTTON_TEXT_COLOR = "#ffffff"

RISK_BUTTON_COLOR = "#c92a2a"
RISK_BUTTON_HOVER_COLOR = "#a61e1e"
RISK_BUTTON_PRESSED_COLOR = "#871919"

ACCENT_COLOR = "#1f4e79"
SUCCESS_COLOR = "#1f7a4d"
INFO_COLOR = "#274c77"
MUTED_COLOR = "#4f5d75"

OUTPUT_BG_COLOR = "#eef5ff"

LIGHT_ENTRY_BG_COLOR = "#f1f3f5"
LIGHT_ENTRY_FG_COLOR = "#1f1f1f"

DARK_ENTRY_BG_COLOR = "#1f1f1f"
DARK_ENTRY_FG_COLOR = "#ffffff"


# Regression & Classification:
CLASSIFICATION_TASK = "Classification"
REGRESSION_TASK = "Regression"

CLASSIFICATION_TARGET = "default_flag"
REGRESSION_TARGET = "repayment_months"

CLASSIFICATION_MODEL_NAME = "Kernel SVM"
REGRESSION_MODEL_NAME = "Decision Tree Regressor"


# Parameter display names:
RANDOM_STATE_LABEL = "Random seed:"
TEST_SIZE_LABEL = "Evaluation split size:"

C_LABEL = "Regularization parameter (C):"
GAMMA_LABEL = "Kernel influence (gamma):"
FIXED_GAMMA_LABEL = "Fixed gamma value:"
CLASS_WEIGHT_LABEL = "Class balance weighting:"

MAX_DEPTH_LABEL = "Tree depth limit:"
MIN_SAMPLES_LEAF_LABEL = "Minimum records per leaf:"


# Dropdown options:
GAMMA_SCALE_OPTION = "scale"
GAMMA_AUTO_OPTION = "auto"
GAMMA_FIXED_OPTION = "fixed"

CLASS_WEIGHT_BALANCED_OPTION = "Balance classes (balanced)"
CLASS_WEIGHT_NONE_OPTION = "Do not balance (none)"


# Input limits:
MAX_TREE_DEPTH = 50
MAX_MIN_SAMPLES_LEAF = 500


# Dropping targets, IDs and leakage columns to not use as feature inputs
DROP_COLS = [
    "default_flag",
    "repayment_months",
    "delinquent_any",
    "delinquent_count",
    "requested_term_months",
    "severity_target",
    "LLAVEVIV",
    "LLAVEHOG",
    "LLAVEMOD",
    "LLAVESDE",
    "FOLIO",
    "HOGAR",
]