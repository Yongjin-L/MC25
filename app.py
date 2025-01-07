import os
from flask import Flask, render_template_string, session, jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>MC25 KIN217 with Mr. Lee</title>
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
    .video-container {
      display: inline-block; 
      position: relative;
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
    #overlay {
      position: absolute;
      top: 50px; 
      left: 10px;
      color: #fff; 
      font-weight: bold; 
      text-shadow: 1px 1px #000;
    }
    #modelStatus {
      margin-top: 10px;
      font-weight: bold;
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
  <h1>MC25 KIN217 with Mr. Lee</h1>

  <!-- 1) Upload THREE files: metadata.js, model.js, weights.bin -->
  <p><strong>Upload your metadata.js, model.js, and weights.bin:</strong></p>
  <input type="file" id="modelFiles" multiple accept=".js,.bin" />
  <div id="modelStatus" style="color:red;"></div>

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

  <!-- 4) Task Summary (hidden until stop) -->
  <div id="summaryContainer" style="display:none;">
    <h2>Task Summary</h2>
    <div id="summaryContent"></div>
  </div>

  <!-- Load TensorFlow.js -->
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>

  <script>
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

    let selectedFiles = [];
    let metadata = null;    // Will store parsed class labels
    let model = null;       // tf.Model or tf.GraphModel
    let inferenceInterval = null;
    let countdownInterval = null;
    let timeCounter = 0;    // in seconds
    let classStats = {};    // { className: { seconds: 0, sumConfidence: 0, count: 0 } }

    modelFilesInput.addEventListener('change', (evt) => {
      selectedFiles = Array.from(evt.target.files);
      modelStatus.textContent = "Files selected. Parsing...";
      modelStatus.style.color = "orange";
    });

    // 1) Parse metadata.js, load model.js + weights.bin
    // We'll do this right before Start Task, if not already loaded.
    async function loadAllFiles(files) {
      // We expect exactly 3 files:
      // 1) metadata.js  2) model.js  3) weights.bin
      // Let's identify them by filename or extension
      let metaFile = null;
      let modelFile = null;
      let weightFile = null;

      for (const f of files) {
        const fname = f.name.toLowerCase();
        if (fname.endsWith("metadata.js")) {
          metaFile = f;
        } else if (fname.endsWith("model.js")) {
          modelFile = f;
        } else if (fname.endsWith(".bin")) {
          weightFile = f;
        }
      }

      if (!metaFile || !modelFile || !weightFile) {
        throw new Error("Please select metadata.js, model.js, and weights.bin!");
      }

      // 1.1) Parse metadata.js for class labels
      metadata = await parseMetadataFile(metaFile);

      // 1.2) Load model (model.js + weights.bin) via tf.io.browserFiles
      // Note: We pass these 2 files to tf.io.browserFiles
      const modelAndWeightsFiles = [modelFile, weightFile];
      try {
        // Attempt graph model first
        const gModel = await tf.loadGraphModel(tf.io.browserFiles(modelAndWeightsFiles));
        return gModel;
      } catch (gErr) {
        console.warn("Graph model load failed, trying layers model...", gErr);
        // Then try layers model
        try {
          const lModel = await tf.loadLayersModel(tf.io.browserFiles(modelAndWeightsFiles));
          return lModel;
        } catch (lErr) {
          console.error("Both graph & layers model load failed:", lErr);
          throw lErr;
        }
      }
    }

    // Parse the metadata.js file content (which we assume is JSON)
    function parseMetadataFile(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (evt) => {
          try {
            const text = evt.target.result;
            // parse it as JSON
            const json = JSON.parse(text);
            resolve(json);
          } catch (err) {
            reject(err);
          }
        };
        reader.onerror = (err) => {
          reject(err);
        };
        reader.readAsText(file);
      });
    }

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

    // 3) Start Task with 5-second countdown
    startTaskButton.addEventListener('click', async () => {
      // If model not loaded yet, load
      if (!model) {
        if (!selectedFiles || selectedFiles.length < 3) {
          alert("Please select metadata.js, model.js, and weights.bin first!");
          return;
        }
        try {
          modelStatus.textContent = "Loading files...";
          modelStatus.style.color = "orange";
          model = await loadAllFiles(selectedFiles);
          modelStatus.textContent = "Model loaded successfully!";
          modelStatus.style.color = "green";
        } catch (err) {
          modelStatus.textContent = "Error loading model files!";
          modelStatus.style.color = "red";
          console.error(err);
          return;
        }
      }

      // Reset stats
      timeCounter = 0;
      classStats = {};
      overlayElem.textContent = "";
      summaryContainer.style.display = "none";

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

      showSummary();
    });

    // Real inference each second
    async function startInference() {
      inferenceInterval = setInterval(async () => {
        timeCounter++;

        // Capture frame
        const inputTensor = tf.browser.fromPixels(camera);
        // Resize if needed (depends on your model)
        const resized = tf.image.resizeBilinear(inputTensor, [224, 224]);
        // Normalize if needed
        const normalized = resized.div(255);
        const batched = normalized.expandDims(0);

        // Predict
        const predictions = model.predict(batched);
        const data = await predictions.data();

        // Identify top class
        let bestIndex = 0;
        let bestScore = data[0];
        for (let i = 1; i < data.length; i++) {
          if (data[i] > bestScore) {
            bestScore = data[i];
            bestIndex = i;
          }
        }
        const confidencePercent = (bestScore * 100).toFixed(1);

        // If metadata.js has "labels" array
        let className = `Class ${bestIndex}`;
        if (metadata && metadata.labels && metadata.labels[bestIndex]) {
          className = metadata.labels[bestIndex];
        }

        overlayElem.textContent = 
          `Time: ${timeCounter}s | ${className} (${confidencePercent}%)`;

        // Track stats
        if (!classStats[className]) {
          classStats[className] = {
            seconds: 0,
            sumConfidence: 0,
            count: 0
          };
        }
        classStats[className].seconds++;
        classStats[className].sumConfidence += parseFloat(confidencePercent);
        classStats[className].count++;

        // Cleanup
        predictions.dispose();
        batched.dispose();
        resized.dispose();
        normalized.dispose();
        inputTensor.dispose();
      }, 1000);
    }

    // 5) Show summary
    function showSummary() {
      const totalTime = timeCounter;
      let html = `<p><strong>Total Task Time:</strong> ${totalTime} seconds</p>`;

      // Each class: Name, average confidence, seconds
      for (const cname in classStats) {
        const stats = classStats[cname];
        const avgConf = (stats.sumConfidence / stats.count).toFixed(1);
        html += `<p><strong>${cname}</strong>: ${stats.seconds}s, Avg Confidence: ${avgConf}%</p>`;
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

if __name__ == "__main__":
    # For local or Render deployment
    import sys
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
