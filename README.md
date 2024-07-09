# Predictive Maintenance Flask Application

This Flask application is designed for predictive analysis using the NASA C-MAPSS Jet Engine Simulated Dataset. It allows users to upload datasets, preprocess data, apply machine learning models, and visualize results. Additionally, it integrates a voice transcription feature and LangChain for natural language processing.

## Features

1. **Dataset Understanding**: Analyze the NASA C-MAPSS dataset for predicting remaining useful life (RUL).
2. **Data Uploading and Preprocessing**: Upload datasets and perform preprocessing tasks.
3. **Machine Learning Models**: Apply various machine learning models for predictive maintenance.
4. **Voice Transcriber Integration**: Use voice input for queries and instructions.
5. **LangChain Integration**: Interpret and analyze textual queries.
6. **User Interface**: Intuitive UI for data exploration, model selection, and result visualization.
7. **Testing and Optimization**: Ensured reliability and performance through rigorous testing.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Flask
- pandas
- scikit-learn
- numpy
- openai
- langchain

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/thississid/Predictive_analysis_flask_app.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Predictive_analysis_flask_app
    ```
3. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Run the Flask application:
    ```sh
    flask run
    ```
2. Open your browser and navigate to:
    ```
    http://127.0.0.1:5000/
    ```

### File Structure

```plaintext
predictive-maintenance-flask-app/
│
├── app/
│   ├── static/
│   ├── templates/
│   ├── __init__.py
│   ├── routes.py
│   ├── forms.py
│   ├── models.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── run.py
