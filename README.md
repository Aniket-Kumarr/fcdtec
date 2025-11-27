# Face Detection AI - Desktop Application

A standalone desktop application for face detection with LLM processing. Built with Python, OpenCV, and Tkinter.

## Features

- 🎯 Real-time face detection using OpenCV
- 📷 Webcam support with live detection
- 🖼️ Image file support (JPG, PNG, BMP, GIF)
- 🤖 LLM-powered analysis (OpenAI integration)
- 🖥️ Native desktop GUI (no browser required)

## Requirements

- Python 3.8 or higher
- Webcam (for webcam mode)
- (Optional) OpenAI API key for LLM features

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

Or using pip3:
```bash
pip3 install -r requirements.txt
```

## Usage

### Basic Usage

Run the application:
```bash
python app.py
```

Or:
```bash
python3 app.py
```

### With OpenAI API (Optional)

To enable AI analysis features, set your OpenAI API key as an environment variable:

**On macOS/Linux:**
```bash
export OPENAI_API_KEY=your_key_here
python app.py
```

**On Windows:**
```cmd
set OPENAI_API_KEY=your_key_here
python app.py
```

Or create a `.env` file (you'll need python-dotenv package):
```
OPENAI_API_KEY=your_key_here
```

The app will work without the API key, but AI analysis features will be limited.

## How to Use

1. **Webcam Mode** (default):
   - Click "Start Webcam" to begin
   - Face detection runs automatically in real-time
   - Click "Analyze with AI" to get insights about detected faces
   - Click "Stop Webcam" when done

2. **Image File Mode**:
   - Select "Image File" radio button
   - Click "Load Image" to select an image file
   - Click "Detect Faces" to analyze the image
   - Click "Analyze with AI" for LLM insights

## Features

- **Real-time Detection**: Automatically detects faces in webcam feed
- **Face Counting**: Shows number of faces detected
- **Visual Feedback**: Green rectangles highlight detected faces
- **AI Analysis**: Get intelligent insights about detected faces (requires API key)

## Troubleshooting

- **Webcam not working**: Make sure your webcam is connected and not being used by another application
- **No faces detected**: Try adjusting lighting or moving closer to the camera
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

## License

MIT
