from flask import Flask, render_template_string, request, redirect, url_for, session

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Local TF.js Model Loader</title>
</head>
<body style="font-family: Arial, sans-serif; text-align:center; margin: 30px;">
  <h1>Local Model Loader + Camera Demo</h1>

  <!-- 1) Model File Inputs -->
  <p>Select your TensorFlow.js model files (JSON + BIN):</p>
  <input type="file" id="modelFiles" multiple accept=".json,.bin" />
  <br/><br/>

  <!-- 2) Open Camera Button -->
  <button id="openCameraButton">Open Camera</button>
  <br/><br/>

  <!-- 3) Video/Inference Area -->
  <div style="display:inline-block; position:relative;">
    <video id="camera" width="640" height="480" autoplay muted style="background:#333"></video>
    <div id="overlay" style="position:absolute; top:0; left:0; color:#fff; font-weight:bold;">
      <!-- We'll display inference info here -->
    </div>
  </div>
  <br/><br/>

  <!-- 4) Start/Stop Demo -->
  <button id="startButton">Start Task</button>
  <button id="stopButton" disabled>Stop Task</button>

  <!-- Script Section -->
  <!-- Load TensorFlow.js -->
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>

  <script>
    let videoElem = null;
    let model = null;
    let inferenceInterval = null;

    const openCameraButton = document.getElementById('openCameraButton');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const camera = document.getElementById('camera');
    const overlay = document.getElementById('overlay');
    const modelFilesInput = document.getElementById('modelFiles');

    // 1) Let user pick model.json & .bin
    // We'll store them in a variable to load after user picks them
    let selectedFiles = [];

    modelFilesInput.addEventListener('change', (evt) => {
      selectedFiles = Array.from(evt.target.files); 
      console.log('Selected files:', selectedFiles);
    });

    // 2) Open Camera
    openCameraButton.addEventListener('click', async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        camera.srcObject = stream;
      } catch (err) {
        alert("Could not access camera. Please allow permissions.");
        console.error(err);
      }
    });

    // 3) Start Task
    startButton.addEventListener('click', async () => {
      // If user hasn't picked files or no model loaded yet, load it
      if (!model) {
        if (!selectedFiles || selectedFiles.length === 0) {
          alert("Please select your model.json and .bin file(s) first.");
          return;
        }
        try {
          // Use tf.io.browserFiles to load them
          model = await loadLocalModel(selectedFiles);
          console.log("Model loaded:", model);
        } catch (err) {
          alert("Error loading model from files.");
          console.error(err);
          return;
        }
      }
      // Start "inference"
      startButton.disabled = true;
      stopButton.disabled = false;
      runInference();
    });

    // 4) Stop Task
    stopButton.addEventListener('click', () => {
      if (inferenceInterval) {
        clearInterval(inferenceInterval);
      }
      startButton.disabled = false;
      stopButton.disabled = true;
      overlay.innerHTML = "";
    });

    // Utility: Load the local model with tf.io.browserFiles
    async function loadLocalModel(files) {
      // You can detect if it's a graph model or layers model
      // For Teachable Machine, it might be a layers model
      // But let's try both. We'll assume it's a graph model first
      // If that fails, we can fallback to a layers model, etc.

      // Try a graph model
      try {
        const model = await tf.loadGraphModel(tf.io.browserFiles(files));
        return model;
      } catch (e1) {
        console.warn("Graph model load failed, trying layers model...", e1);
        // Try layers model
        try {
          const model = await tf.loadLayersModel(tf.io.browserFiles(files));
          return model;
        } catch (e2) {
          console.error("Both graph model & layers model load failed:", e2);
          throw e2;
        }
      }
    }

    // Mock or Real Inference
    function runInference() {
      // If we had real data from the camera, we'd do something like:
      // const predictions = model.predict(processVideoFrame(camera));
      // Instead, let's just mock it every 1 second
      inferenceInterval = setInterval(() => {
        // For demonstration, random "Confidence" 0-100
        const randomConfidence = Math.floor(Math.random() * 101);
        overlay.innerHTML = `Mock Inference: ${randomConfidence}% confidence`;
      }, 1000);
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    # Renders a single-page app with file inputs, camera, etc.
    return render_template_string(INDEX_HTML)

# (Optional) A summary route if you want a separate page:
@app.route("/summary")
def summary():
    return "<h1>Summary Page</h1><p>Not implemented yet.</p>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
