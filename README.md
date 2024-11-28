# Multimodal Emotion Analysis System

A robust system that combines facial emotion recognition with text sentiment analysis to provide comprehensive emotional insights and personalized recommendations.

## 🌟 Features

- **Text Analysis**
  - Sentiment classification (positive, negative, neutral)
  - Emotion detection (joy, sadness, anger, fear, love, surprise)
  - Tweet and general text support

- **Facial Analysis**
  - Real-time facial emotion detection
  - Region-based facial analysis for improved accuracy
  - Support for multiple faces in a single image
  - Batch processing capabilities

- **AI Recommendations**
  - Personalized insights using GPT-4
  - Context-aware emotional state analysis
  - Actionable recommendations based on combined analysis

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- Flask
- OpenAI API key
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-analysis-system.git
cd emotion-analysis-system
```

2. Set up environment variables:
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

3. Start the server:
```bash
python app.py
```

4. Access the web interface at `http://localhost:5000`

## 💡 Usage

### Text Analysis
```python
from tweet_analyzer import TweetAnalyzer

analyzer = TweetAnalyzer()
results = analyzer.analyze_tweet("Your text here")
```

### Image Analysis
```python
from batch_classifier import BatchImageClassifier

classifier = BatchImageClassifier("path/to/model.h5")
results = classifier.process_image_folder("input_folder", "output_folder")
```

## 🏗️ Project Structure

```
.
├── app.py              # Main Flask application
├── tweet_analyzer.py   # Text analysis module
├── batch_classifier.py # Image analysis module
├── models/            # Pre-trained models
├── templates/         # HTML templates
└── uploads/          # Temporary file storage
```

## 📊 Model Performance

- Text Sentiment Analysis: 71% accuracy
- Facial Emotion Recognition: 64% accuracy
- Support for 8 distinct emotions

## 🔒 Security Considerations

- Maximum file size: 8MB
- Supported image formats: PNG, JPG, JPEG
- Automatic file cleanup after processing
- GPU memory optimization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- OpenAI for GPT-4 integration
- Contributors and maintainers

## 📞 Contact

Your Name - bojalil.david@gmail.com
Project Link: [https://github.com/DavidBo9/EmotiAISourceCode](https://github.com/DavidBo9/EmotiAISourceCode)