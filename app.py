import os
from flask import Flask, render_template_string

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>MC25 KIN217 with Mr. Lee - Overlay on Camera</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      text-align: center;
      margin: 30px;
      background: #f0f0f0;
    }
    h1 {
      margin-bottom: 20px;
    }
    .btn {
      padding: 10px 20px;
      font-size: 1rem;
      margin: 5px;
    }
    /* Container to hold the video and overlay absolutely */
    .video-container {
      position: relative;
      display: inline-block;
      background: #000; /* fallback if camera isn't open */
    }
    /* The actual video feed */
    #camera {
      width: 640px;
      height: 480px;
      background: #222; /* fallback background if no camera feed */
      z-index: 1;
    }
    /* The overlay that goes on top of the video */
    #overlay {
      position: absolute;
      top: 10px;
      left: 10px;
      color: #fff;
      font-weight: bold;
      text-shadow: 1px 1px #000;
      z-index: 2; /* ensure it's above the video */
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
  <h1>MC25 KIN217 with Mr. Lee - Overlay on Camera</h1>

  <!-- 1) Upload your model.json & weights.bin (2D input) -->
  <p><strong>Upload model.json & weights.bin (and optional metadata.json):</strong></p>
  <input type="file" id="modelFiles" multiple accept=".json,.bin" />
  <div id="modelStatus" style="color:red;"></div>

  <!-- 2) Buttons: Open Camera / Start Task / Stop Task -->
  <div>
    <button class="btn" id="openCameraButton">Open Camera</button>
    <button class="btn" id="startTaskButton">Start Task</button>
    <button class="btn" id="stopTaskButton" disabled>Stop Task</button>
  </div>

  <!-- 3) Video container with overlay text on top -->
  <div class="video-container">
    <video id="camera" autoplay muted playsinline></video>
    <div id="overlay"></div>
  </div>

  <!-- 4) Summary after Stop Task -->
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
    const overlayElem = document.getElementById('overlay');
    const summaryContainer = document.getElementById('summaryContainer');
    const summaryContent = document.getElementById('summaryContent');

    let selectedFiles = [];
    let model = null;
    let metadata = null; // optional for class labels
    let inferenceInterval = null;
    let timeCounter = 0;
    let classStats = {}; // { className: { seconds, sumConfidence, count }}

    modelFilesInput.addEventListener('change', (evt) => {
      selectedFiles = Array.from(evt.target.files);
      modelStatus.textContent = "Files selected. Ready to load...";
      modelStatus.style.color = "orange";
    });

    // Optional: open camera feed (user must allow)
    openCameraButton.addEventListener('click', async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        camera.srcObject = stream;
      } catch (err) {
        alert("Could not access camera. Please allow permissions if you want a live feed.");
        console.error(err);
      }
    });

    // Load model + (optional) metadata
    async function loadModelAndWeights(files) {
      let modelFile = null;
      let weightsFile = null;
      let metaFile = null;

      for (const f of files) {
        const fname = f.name.toLowerCase();
        if (fname.endsWith("model.json")) {
          modelFile = f;
        } else if (fname.endsWith(".bin")) {
          weightsFile = f;
        } else if (fname.endsWith("metadata.json")) {
          metaFile = f;
        }
      }
      if (!modelFile || !weightsFile) {
        throw new Error("Please select model.json and weights.bin!");
      }

      // If there's metadata, parse it
      if (metaFile) {
        metadata = await parseMetadataFile(metaFile);
      }

      // Attempt graph or layers
      const modelAndWeights = [modelFile, weightsFile];
      try {
        const gModel = await tf.loadGraphModel(tf.io.browserFiles(modelAndWeights));
        return gModel;
      } catch (gErr) {
        console.warn("Graph model load failed, trying layers model...", gErr);
        try {
          const lModel = await tf.loadLayersModel(tf.io.browserFiles(modelAndWeights));
          return lModel;
        } catch (lErr) {
          console.error("Both graph & layers model load failed:", lErr);
          throw lErr;
        }
      }
    }

    // Parse metadata.json if present
    function parseMetadataFile(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (evt) => {
          try {
            const json = JSON.parse(evt.target.result);
            resolve(json);
          } catch (err) {
            reject(err);
          }
        };
        reader.onerror = (err) => reject(err);
        reader.readAsText(file);
      });
    }

    // Start Task
    startTaskButton.addEventListener('click', async () => {
      if (!model) {
        if (!selectedFiles || selectedFiles.length < 2) {
          alert("Please select model.json and weights.bin!");
          return;
        }
        try {
          modelStatus.textContent = "Loading model...";
          modelStatus.style.color = "orange";
          model = await loadModelAndWeights(selectedFiles);
          modelStatus.textContent = "Model loaded successfully!";
          modelStatus.style.color = "green";
        } catch (err) {
          modelStatus.textContent = "Error loading model!";
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

      startTaskButton.disabled = true;
      stopTaskButton.disabled = false;

      // Inference loop every 1 second
      inferenceInterval = setInterval(() => {
        timeCounter++;

        // Create or fetch a [1, 14739] vector for your 2D model
        // For demonstration, use random data. Replace with your real features if needed.
        const inputTensor = tf.randomNormal([1, 14739]);

        // Predict
        const predictions = model.predict(inputTensor);
        predictions.data().then((data) => {
          // Suppose model outputs 2 classes (like your snippet)
          let bestIndex = 0;
          let bestScore = data[0];
          for (let i = 1; i < data.length; i++) {
            if (data[i] > bestScore) {
              bestScore = data[i];
              bestIndex = i;
            }
          }
          const confidencePercent = (bestScore * 100).toFixed(1);

          // If you have metadata.labels
          let className = `Class ${bestIndex}`;
          if (metadata && metadata.labels && metadata.labels[bestIndex]) {
            className = metadata.labels[bestIndex];
          }

          // Overlay text
          overlayElem.textContent = `Time: ${timeCounter}s | ${className} (${confidencePercent}%)`;

          // Update stats
          if (!classStats[className]) {
            classStats[className] = { seconds: 0, sumConfidence: 0, count: 0 };
          }
          classStats[className].seconds++;
          classStats[className].sumConfidence += parseFloat(confidencePercent);
          classStats[className].count++;
        });

        // Clean up
        predictions.dispose();
        inputTensor.dispose();
      }, 1000);
    });

    // Stop Task
    stopTaskButton.addEventListener('click', () => {
      if (inferenceInterval) {
        clearInterval(inferenceInterval);
      }
      startTaskButton.disabled = false;
      stopTaskButton.disabled = true;
      overlayElem.textContent = "";

      showSummary();
    });

    // Show Summary
    function showSummary() {
      let html = `<p><strong>Total Task Time:</strong> ${timeCounter} seconds</p>`;
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
    import sys
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
