# app.py
import os
import sys
import time

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QMessageBox, QFrame, QProgressBar
)
from PyQt6.QtGui import QFont, QPixmap, QIcon
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal

# Machine Learning Imports
from transformers import pipeline
import torch

# --- Helper Function for File Paths (for PyInstaller compatibility) ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller. """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # Fallback for normal execution
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Project Configuration ---
MODEL_DIR = resource_path("best_model")
IMAGE_DIR = resource_path("emoji_images")

ID2IMAGE = {
    0: "heart.png", 1: "heart_eyes.png", 2: "joy.png", 3: "two_hearts.png", 4: "fire.png",
    5: "blush.png", 6: "sunglasses.png", 7: "sparkles.png", 8: "blue_heart.png", 9: "kissing_heart.png",
    10: "camera.png", 11: "us_flag.png", 12: "sun.png", 13: "purple_heart.png", 14: "wink.png",
    15: "hundred.png", 16: "grin.png", 17: "tree.png", 18: "camera_flash.png", 19: "wink_tongue.png"
}

# --- Splash Screen: Shows while the heavy AI model is loading ---
class SplashScreen(QWidget):
    """ A simple, modern splash screen with a logo and progress bar. """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loading Application")
        self.setFixedSize(400, 250)
        # Frameless window that stays on top
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._setup_ui()

    def _setup_ui(self):
        """Builds the UI elements for the splash screen."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        frame = QFrame()
        frame.setObjectName("splash_frame")
        layout.addWidget(frame)

        splash_layout = QVBoxLayout(frame)
        
        logo_label = QLabel()
        logo_path = os.path.join(IMAGE_DIR, "splash_logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        splash_layout.addWidget(logo_label)

        self.status_label = QLabel("Initializing, please wait...")
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        splash_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setTextVisible(False)
        splash_layout.addWidget(self.progress_bar)

        self.setStyleSheet("""
            QFrame#splash_frame { background-color: #FFFFFF; border-radius: 15px; border: 1px solid #E0E5E9; }
            QLabel#status_label { font-size: 14px; color: #5A6978; }
            QProgressBar { height: 8px; border: none; background-color: #E9EDF1; border-radius: 4px; margin: 0 20px; }
            QProgressBar::chunk { background-color: #007AFF; border-radius: 4px; }
        """)
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)

# --- Model Loader: Runs model loading in a separate thread ---
class ModelLoader(QThread):
    """
    Loads the model in a background thread to prevent the UI from freezing.
    Emits signals to update the splash screen progress and to indicate completion.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def run(self):
        try:
            self.progress.emit(10)
            time.sleep(0.5) # Simulate initial steps for a smoother progress bar
            self.progress.emit(30)
            
            model_pipeline = pipeline(
                "text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR,
                top_k=None, device=-1 # Force CPU for max compatibility
            )
            
            self.progress.emit(80)
            time.sleep(0.5) # Simulate finalization
            self.progress.emit(100)
            self.finished.emit(model_pipeline) # Send the loaded model back
        except Exception as e:
            self.finished.emit(e) # Send the error back if loading fails

# --- Main Application Window ---
class EmojiPredictorApp(QWidget):
    """ The main GUI for the Tweet Emoji Predictor application. """
    def __init__(self, model_pipeline):
        super().__init__()
        self.model_pipeline = model_pipeline
        self.first_prediction = True

        self.setWindowTitle("Tweet Emoji Predictor")
        self.setGeometry(100, 100, 620, 720) 
        
        icon_path = os.path.join(IMAGE_DIR, "app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        """Creates and arranges all the widgets in the main window."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(25, 20, 25, 20)
        main_layout.setSpacing(15)

        # --- Header ---
        title_label = QLabel("Tweet Emoji Predictor")
        title_label.setObjectName("title_label")
        main_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        description_label = QLabel(
            "Unlock the sentiment! This app uses a fine-tuned BERTweet model to predict the top 3 most likely emojis for your text. Just type or paste your tweet below and hit submit!"
        )
        description_label.setObjectName("description_label")
        description_label.setWordWrap(True)
        main_layout.addWidget(description_label)

        # --- Input Card ---
        input_frame = QFrame()
        input_frame.setObjectName("card_frame")
        input_layout = QVBoxLayout(input_frame)
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Feeling absolutely ecstatic about this project!")
        input_layout.addWidget(self.text_input)
        predict_button = QPushButton("✨ Predict Emojis")
        predict_button.clicked.connect(self.handle_prediction)
        input_layout.addWidget(predict_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(input_frame)

        # --- Examples ---
        examples_label = QLabel("Or try an example:")
        examples_label.setObjectName("example_title_label")
        main_layout.addWidget(examples_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        examples_layout = QVBoxLayout()
        examples_layout.setSpacing(6)
        example_tweets = [
            "That movie was so funny, I was in tears",
            "Missing the beach and sunshine right now",
            "Feeling absolutely ecstatic about this project!"
        ]
        for tweet_text in example_tweets:
            btn = QPushButton(tweet_text)
            btn.setObjectName("example_button")
            btn.clicked.connect(lambda checked, text=tweet_text: self.load_example(text))
            examples_layout.addWidget(btn)
        main_layout.addLayout(examples_layout)

        # --- Results Card ---
        results_frame = QFrame()
        results_frame.setObjectName("card_frame")
        card_layout = QVBoxLayout(results_frame)
        self.results_title = QLabel("Top 3 Predictions")
        self.results_title.setObjectName("results_title_label")
        card_layout.addWidget(self.results_title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.placeholder_label = QLabel("Your predictions will appear here ✨")
        self.placeholder_label.setObjectName("placeholder_label")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.placeholder_label)

        self.results_container = QWidget()
        results_holder_layout = QHBoxLayout(self.results_container)
        self.results_container.hide()
        
        self.emoji_image_labels = []
        self.emoji_percent_labels = []
        for i in range(3):
            single_result_layout = QVBoxLayout()
            single_result_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            emoji_image_label = QLabel()
            emoji_image_label.setFixedSize(QSize(56, 56))
            self.emoji_image_labels.append(emoji_image_label)
            single_result_layout.addWidget(emoji_image_label)
            emoji_percent_label = QLabel()
            emoji_percent_label.setObjectName("percentage_label")
            self.emoji_percent_labels.append(emoji_percent_label)
            single_result_layout.addWidget(emoji_percent_label)
            results_holder_layout.addLayout(single_result_layout)
        card_layout.addWidget(self.results_container)
        main_layout.addWidget(results_frame)
        main_layout.addStretch(1)

    def apply_styles(self):
        """Sets the CSS stylesheet for the application."""
        self.setStyleSheet("""
            QWidget { font-family: 'Segoe UI', Arial, sans-serif; background-color: #F4F6F8; }
            QLabel#title_label { font-size: 32px; font-weight: bold; color: #1E2A3A; margin-bottom: 5px; }
            QLabel#description_label { font-size: 14px; color: #5A6978; margin-bottom: 8px; }
            QFrame#card_frame { background-color: #FFFFFF; border: 1px solid #E0E5E9; border-radius: 12px; padding: 18px; }
            QTextEdit { background-color: #FDFDFD; border: 1px solid #D8DEE3; border-radius: 8px; padding: 12px; font-size: 16px; color: #333333; min-height: 80px; }
            QPushButton { background-color: #007AFF; color: white; font-size: 16px; font-weight: bold; border: none; border-radius: 10px; padding: 11px 24px; min-width: 180px; }
            QPushButton:hover { background-color: #0056b3; }
            QLabel#example_title_label { color: #5A6978; font-size: 12px; font-weight: bold; margin-top: 5px; }
            QPushButton#example_button { background-color: #E9EDF1; color: #3A4754; font-size: 13px; font-weight: normal; padding: 9px; border-radius: 8px; }
            QPushButton#example_button:hover { background-color: #DDE2E6; }
            QLabel#results_title_label { font-size: 20px; font-weight: bold; color: #1E2A3A; margin-bottom: 10px; }
            QLabel#percentage_label { font-size: 26px; font-weight: bold; color: #007AFF; }
            QLabel#placeholder_label { font-size: 16px; color: #8A99A8; padding: 30px; }
        """)

    def load_example(self, text):
        """Fills the text box with an example and triggers a prediction."""
        self.text_input.setText(text)
        self.handle_prediction()

    def handle_prediction(self):
        """Processes the input text and displays the top 3 emoji predictions."""
        input_text = self.text_input.toPlainText().strip()
        if not input_text:
            QMessageBox.warning(self, "Input Error", "Please enter some text to predict.")
            return

        try:
            # On the first prediction, switch from the placeholder to the results view
            if self.first_prediction:
                self.placeholder_label.hide()
                self.results_container.show()
                self.first_prediction = False

            raw_predictions = self.model_pipeline(input_text)[0]
            
            # Process scores to get the top 3 and normalize them to sum to 100%
            processed_predictions = [{'id': int(p['label'].split('_')[1]), 'score': p['score']} for p in raw_predictions]
            top3_predictions = sorted(processed_predictions, key=lambda x: x['score'], reverse=True)[:3]
            sum_top3_scores = sum(p['score'] for p in top3_predictions)

            # Update the UI with the results
            for i in range(3):
                if i < len(top3_predictions):
                    prediction = top3_predictions[i]
                    percentage = (prediction['score'] / sum_top3_scores) * 100 if sum_top3_scores > 0 else 0
                    emoji_filename = ID2IMAGE.get(prediction['id'])
                    
                    if emoji_filename:
                        emoji_path = os.path.join(IMAGE_DIR, emoji_filename)
                        if os.path.exists(emoji_path):
                            pixmap = QPixmap(emoji_path)
                            scaled_pixmap = pixmap.scaled(self.emoji_image_labels[i].size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                            self.emoji_image_labels[i].setPixmap(scaled_pixmap)
                        else: self.emoji_image_labels[i].setText("❓")
                    else: self.emoji_image_labels[i].setText("❓")

                    self.emoji_percent_labels[i].setText(f"{percentage:.1f}%")
                else:
                    # Clear any unused result slots
                    self.emoji_image_labels[i].clear()
                    self.emoji_percent_labels[i].setText("")

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred during prediction:\n{e}")

# --- Main Execution Block ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    
    # This global variable is a simple way to hold a reference to the main window
    # after the splash screen closes.
    main_window = None

    def on_model_loaded(result):
        """Callback function that runs when the model loader thread finishes."""
        global main_window
        splash.close()
        
        if isinstance(result, Exception):
            QMessageBox.critical(None, "Fatal Error", f"Could not load the AI model:\n{result}")
            sys.exit(1)
        else:
            # Model loaded successfully, create and show the main application window
            main_window = EmojiPredictorApp(model_pipeline=result)
            main_window.show()

    # Start the model loading in the background
    loader = ModelLoader()
    loader.progress.connect(splash.update_progress)
    loader.finished.connect(on_model_loaded)
    loader.start()
    
    sys.exit(app.exec())