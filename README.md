# Tweet Emoji Predictor 🤖✨

A user-friendly desktop application that uses a pre-trained BERTweet model to predict the top 3 most likely emojis for a given text.

-   **AI-Powered Predictions**: Leverages a fine-tuned BERTweet model to analyze text sentiment.
-   **Interactive UI**: A clean and modern interface built with PyQt6.
-   **Responsive Loading**: A professional splash screen informs the user while the model loads in the background.
-   **Easy to Use**: Simple text input, example buttons, and clear, visual results.
-   **Standalone Application**: Can be packaged into a single executable file for easy distribution.

## Setup and Installation

This project uses a Conda environment to manage complex dependencies, especially for PyTorch.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/tweet-emoji-predictor.git](https://github.com/your-username/tweet-emoji-predictor.git)
    cd tweet-emoji-predictor
    ```

2.  **Create the Conda environment:**
    ```bash
    conda create --name emoji_app python=3.11
    conda activate emoji_app
    ```

3.  **Install PyTorch (CPU Version):**
    This specific command is crucial for compatibility.
    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```

4.  **Install the remaining packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the environment is set up, run the application with:
```bash
python app.py
```

## Project Structure

```
.
├── app.py              # Main application script with all the GUI and logic
├── best_model/         # Folder containing the pre-trained model files
├── emoji_images/       # Folder for all emoji PNGs and UI icons
├── requirements.txt    # Python packages needed for the project
├── .gitignore          # Specifies files for Git to ignore
└── README.md           # This file
```

## Building the Executable

To package the application into a single `.exe` file, use PyInstaller. First, make sure you have an `app_icon.ico` file in the root directory.

```bash
pyinstaller --name "Emoji_Predictor" --onefile --windowed --icon="app_icon.ico" --add-data "best_model;best_model" --add-data "emoji_images;emoji_images" app.py
```
The final executable will be located in the `dist/` folder.