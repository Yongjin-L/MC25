# Pose Duration Tracker with Mr. Lee

A web application for tracking pose durations using Teachable Machine pose models. This tool allows users to load custom pose recognition models and measure the time spent in different poses. The code is developed with Claue 3.7 Sonnet and GPT-4o.

**[https://mc25.onrender.com/](https://mc25.onrender.com/)**

## Overview

This application enables users to:
- Load Teachable Machine pose models via URL
- Test webcam functionality
- Track and time different poses in real-time
- View pose recognition probabilities via live charts
- Generate summaries of pose durations after each session

Perfect for physical education, dance instruction, yoga practice, or any activity requiring pose tracking and duration measurement.

## Features

- **Model Loading**: Load any Teachable Machine pose model via URL
- **Webcam Integration**: Test and use webcam for pose detection
- **Real-time Visualization**: 
  - Live pose skeleton overlay
  - Real-time probability bar chart
  - Status feedback for pose detection quality
- **Duration Tracking**: Automatic tracking of time spent in each detected pose
- **Summary Reporting**: End-of-session summary with total durations by pose
- **Responsive Design**: Works on various screen sizes

## Requirements

- A modern web browser (Chrome, Firefox, Safari, Edge)
- Webcam access
- Internet connection (for loading required libraries)
- A Teachable Machine pose model URL

## Usage Instructions

1. **Load Model**:
   - Enter a valid Teachable Machine pose model URL in the input field
   - Click "Check Model URL" to validate and load the model
   - Example URL format: `https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/`

2. **Test Your Webcam**:
   - Click "Test Webcam" to ensure your camera is working correctly
   - Click "Stop Test" when finished

3. **Start the Task**:
   - Click "Start Task" to begin pose tracking
   - A 5-second countdown will appear before tracking starts
   - The application will display:
     - Current pose class
     - Confidence percentage
     - Elapsed time
     - Visual feedback on pose quality

4. **End the Task**:
   - Click "End Task" when you're finished
   - A summary will display showing:
     - Total session duration
     - Bar chart of time spent in each detected pose

5. **Restart**:
   - Click "Restart" to begin a new session

## Creating a Teachable Machine Pose Model

If you don't have a pose model URL, you can create one:

1. Visit [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Select "New Project" and choose "Pose Project"
3. Create classes for each pose you want to track
4. Train your model using the webcam
5. Export your model and choose "Tensorflow.js"
6. Select "Upload my model" or "Get shareable link"
7. Copy the URL provided to use in this application

## Technical Details

This application uses:
- Flask (Python web framework)
- TensorFlow.js
- Teachable Machine Pose library
- Chart.js for data visualization

## Customization

You can modify the application by editing the HTML and JavaScript in the `HTML_PAGE` variable in the Flask application file.

## Troubleshooting

- **Model Not Loading**: Ensure the URL is correct and includes the trailing slash
- **Webcam Not Working**: Check browser permissions and ensure no other application is using the webcam
- **Poor Pose Detection**: Ensure good lighting and clear visibility of your full body in the camera frame

## License

MIT License
Copyright (c) 2025 Yongjin Lee

## Acknowledgments

- [Teachable Machine](https://teachablemachine.withgoogle.com/) by Google
- [TensorFlow.js](https://www.tensorflow.org/js)
- [Chart.js](https://www.chartjs.org/)
