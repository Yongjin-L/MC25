import os
from flask import Flask, render_template_string, session, jsonify, request, redirect, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Local TF.js Model Loader with Real Inference</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      text-align: center;
      margin: 30px;
    }
    h1 {
      margin-bottom: 20px;
    }
    .btn {
      padding: 10px 20px;
      font-size: 1rem;
      margin: 5px;
    }
    #camera {
      background: #333;
    }
    #overlay {
      position: absolute;
      top: 10px; 
      left: 10px;
      color: #fff; 
      font-weight: bold; 
      text-shadow: 1px 1px #000;
    }
    #countdown {
      position: absolute;
      top: 10px;
      left: 10px;
      font-size: 2em;
      color: #FF0000;
      font-weight: bold;
      text-shadow: 1px 1px #000;
    }
    .video-container {
      display: inline-block; 
      position: relative;
    }
    #modelStatus {
      color: red; 
      font-weight: bold; 
      margin-bottom: 10px;
    }
    #summaryContainer {
      margin-top: 30px;
      text-align: left;
      display: inline-block;
    }
    #summaryContainer p {
      margin: 5px 0;
    }
  </style>
</head>

<body>
  <h1>Local TF.js Model Loader + Real Predictions</h1>

  <!-- 1) Model Files (JSON + BIN) -->
  <div>
    <label><strong>Upload your model.json and .bin:</strong></label><br/>
    <input type="file" id="modelFiles" multiple accept=".json,.bin" />
  </div>
  <div id="modelStatus"></div>

  <!-- 2) Buttons to open camera / start / stop -->
  <button class="btn" id="openCameraButton">Open Camera</button>
  <button class="btn" id="startTaskButton">Start Task</button>
  <button class="btn" id="stopTaskButton" disabled>Stop Task</button>
  <br/><br/>

  <!-- 3) Video + Overlay Container -->
  <div class="video-container">
    <video id="camera" width="640" height="480" autoplay muted playsinline></video>
    <div id="countdown"></div>
    <div id="overlay"></div>
  </div>
  <br/>

  <!-- 4) Optional: Summary Container (hidden until stop) -->
  <div id="summaryContainer" style="display:none;">
    <h2>Task Summary</h2>
    <div id="summaryContent"></div>
  </div>

  <!-- Load TensorFlow.js -->
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
  <!-- 
    If your model requires specific TF.js converters or additional libs,
    import them here as well.
  -->

  <script>
    // HTML Elements
    const modelFilesInput = document.getElementById('modelFiles');
    const modelStatus = document.getElementById('modelStatus');
    const openCameraButton = document.getElementById('openCameraButton');
    const startTaskButton = document.getElementById('startTaskButton');
    const stopTaskButton = document.getElementById('stopTaskButton');
    const camera = document.getElementById('camera');
    const countdownElem = document.getElementById('countdown');
    const overlayElem = document.getElementById('overlay');
    const summaryContainer = document.getElementById('summaryContainer');
    const summaryContent = document.getElementById('summaryContent');

    // Variables
    let model = null;
    let selectedFiles = [];
    let countdownInterval = null;
    let inferenceInterval = null;
    let timeCounter = 0;           // total time in seconds while task is running
    let classStats = {};          // { className: { seconds: 0, sumConfidence: 0, count: 0 } }

    // 1) Selecting local model files
    modelFilesInput.addEventListener('change', (evt) => {
      selectedFiles = Array.from(evt.target.files); 
      console.log("Selected files:", selectedFiles);
      modelStatus.textContent = "Model files selected. (Not yet loaded)";
      modelStatus.style.color = "blue";
    });

    // 2) Open camera
    openCameraButton.addEventListener('click', async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        camera.srcObject = stream;
      } catch (err) {
        alert("Could not access camera. Please allow permissions.");
        console.error(err);
      }
    });

    // 3) Start Task (with 5-second countdown, then real inference)
    startTaskButton.addEventListener('click', async () => {
      // If model not loaded yet, attempt to load
      if (!model) {
        if (!selectedFiles || selectedFiles.length === 0) {
          alert("Please select your model.json and .bin files first!");
          return;
        }
        try {
          modelStatus.textContent = "Loading model...";
          modelStatus.style.color = "orange";
          model = await loadLocalModel(selectedFiles);
          modelStatus.textContent = "Model loaded successfully!";
          modelStatus.style.color = "green";
        } catch (err) {
          console.error("Model loading error:", err);
          modelStatus.textContent = "Error loading model!";
          modelStatus.style.color = "red";
          return;
        }
      }

      // Reset stats
      timeCounter = 0;
      classStats = {};
      overlayElem.textContent = "";
      summaryContainer.style.display = "none"; // hide summary if previously shown

      // 5-second countdown
      let countdownVal = 5;
      countdownElem.textContent = countdownVal;
      startTaskButton.disabled = true;
      stopTaskButton.disabled = true;

      countdownInterval = setInterval(() => {
        countdownVal--;
        if (countdownVal > 0) {
          countdownElem.textContent = countdownVal;
        } else {
          clearInterval(countdownInterval);
          countdownElem.textContent = "";
          stopTaskButton.disabled = false;
          // Start real-time inference
          startInference();
        }
      }, 1000);
    });

    // 4) Stop Task
    stopTaskButton.addEventListener('click', () => {
      if (inferenceInterval) {
        clearInterval(inferenceInterval);
      }
      startTaskButton.disabled = false;
      stopTaskButton.disabled = true;
      overlayElem.textContent = "";

      // Show summary
      showSummary();
    });

    // --- REAL INFERENCE LOGIC EVERY 1 SECOND ---
    async function startInference() {
      inferenceInterval = setInterval(async () => {
        timeCounter++;

        // 1) Capture current video frame as a tensor
        const inputTensor = tf.browser.fromPixels(camera);
        // For Teachable Machine models, you often need to resize / normalize
        // Adjust these steps to your model's expected input shape.
        const resized = tf.image.resizeBilinear(inputTensor, [224, 224]);
        const normalized = resized.div(255);
        const batched = normalized.expandDims(0);

        // 2) Predict
        // If your model is a layers model with "predict":
        const predictions = model.predict(batched);
        // predictions shape might be [1, NCLASSES]
        
        // 3) Get top class and confidence
        const data = await predictions.data();
        // e.g., data might be [0.1, 0.7, 0.2] for 3 classes
        let bestIndex = 0;
        let bestScore = data[0];
        for (let i = 1; i < data.length; i++) {
          if (data[i] > bestScore) {
            bestScore = data[i];
            bestIndex = i;
          }
        }
        const confidencePercent = (bestScore * 100).toFixed(1);
        
        // OPTIONAL: If you know your class labels
        // Hardcode or fetch from somewhere. 
        // For example: 
        const classLabels = ["Class A", "Class B", "Class C"]; 
        // If your model has more classes, extend it accordingly.

        const predictedClassName = classLabels[bestIndex] || `Class ${bestIndex}`;

        // 4) Update overlay
        overlayElem.textContent = 
          `Time: ${timeCounter}s | ${predictedClassName} (${confidencePercent}%)`;

        // 5) Track stats
        if (!classStats[predictedClassName]) {
          classStats[predictedClassName] = {
            seconds: 0,
            sumConfidence: 0,
            count: 0
          };
        }
        classStats[predictedClassName].seconds++;
        classStats[predictedClassName].sumConfidence += parseFloat(confidencePercent);
        classStats[predictedClassName].count++;

        // Cleanup
        batched.dispose();
        resized.dispose();
        normalized.dispose();
        inputTensor.dispose();
        predictions.dispose();
      }, 1000);
    }

    // --- LOADING LOCAL MODEL FILES USING tf.io.browserFiles ---
    async function loadLocalModel(files) {
      // Try graph model first
      try {
        const gModel = await tf.loadGraphModel(tf.io.browserFiles(files));
        return gModel;
      } catch (gErr) {
        console.warn("Graph model load failed, trying layers model...", gErr);
        // Then try layers model
        try {
          const lModel = await tf.loadLayersModel(tf.io.browserFiles(files));
          return lModel;
        } catch (lErr) {
          console.error("Both graph/layers load failed", lErr);
          throw lErr;
        }
      }
    }

    // --- SHOW SUMMARY AFTER STOP ---
    function showSummary() {
      // Calculate total time
      const totalTime = timeCounter; // in seconds

      // Build a summary table or text
      let html = `<p><strong>Total Task Time:</strong> ${totalTime} seconds</p>`;
      html += "<p><strong>Class-by-Class:</strong></p>";
      
      // For each class in classStats
      for (const className in classStats) {
        const stats = classStats[className];
        const avgConfidence = 
          (stats.sumConfidence / stats.count).toFixed(1) || 0;
        html += `
          <p>
            <strong>${className}</strong>: 
            ${stats.seconds} s, 
            Avg Confidence: ${avgConfidence}%
          </p>`;
      }

      summaryContent.innerHTML = html;
      summaryContainer.style.display = "block";
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

# (Optional) If you want a summary route or other endpoints, add here.

if __name__ == "__main__":
    # For local testing or deployment on Render (with minor tweaks for the port).
    import sys
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
