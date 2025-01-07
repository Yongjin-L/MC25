import os
from flask import Flask, render_template_string

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Pose Detection Exercise Assistant</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      text-align: center;
      margin: 30px;
      background: #f0f0f0;
    }
    h1 {
      margin-bottom: 10px;
    }
    h2 {
      margin-top: 0;
      margin-bottom: 20px;
      font-weight: normal;
      color: #555;
    }
    .btn {
      padding: 10px 20px;
      font-size: 1rem;
      margin: 5px;
      cursor: pointer;
    }
    .video-container {
      position: relative;
      display: inline-block;
      background: #000;
    }
    #camera {
      width: 640px;
      height: 480px;
      background: #222;
      z-index: 1;
    }
    #overlay {
      position: absolute;
      top: 10px;
      left: 10px;
      color: #fff;
      font-weight: bold;
      text-shadow: 1px 1px #000;
      z-index: 2;
    }
    #modelStatus {
      margin-top: 10px;
      font-weight: bold;
      color: red;
    }
    #countdown {
      position: absolute;
      top: 10px;
      left: 10px;
      font-size: 2em;
      color: #FF0000;
      font-weight: bold;
      text-shadow: 1px 1px #000;
      z-index: 3;
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
  <h1>Pose Detection Exercise Assistant</h1>
  <h2>MC25 KIN217 with Mr. Lee</h2>

  <div>
    <input type="file" id="modelFiles" multiple accept=".json,.bin" />
    <button class="btn" id="modelLoadButton">Model Load</button>
    <div id="modelStatus"></div>
  </div>
  <div>
    <button class="btn" id="openCameraButton">Open Camera</button>
    <button class="btn" id="startTaskButton">Start Task</button>
    <button class="btn" id="stopTaskButton" disabled>Stop Task</button>
  </div>

  <div class="video-container">
    <video id="camera" autoplay muted playsinline></video>
    <div id="countdown"></div>
    <div id="overlay"></div>
  </div>

  <div id="summaryContainer" style="display:none;">
    <h2>Task Summary</h2>
    <div id="summaryContent"></div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
  <script>
    const modelFilesInput = document.getElementById('modelFiles');
    const modelLoadButton = document.getElementById('modelLoadButton');
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
    let model = null;
    let metadata = null;
    let inferenceInterval = null;
    let countdownInterval = null;
    let timeCounter = 0;
    let classStats = {};

    modelFilesInput.addEventListener('change', e => {
      selectedFiles = Array.from(e.target.files);
      modelStatus.textContent = "Files selected.";
      modelStatus.style.color = "orange";
    });

    modelLoadButton.addEventListener('click', async () => {
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
      }
    });

    openCameraButton.addEventListener('click', async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        camera.srcObject = stream;
      } catch (err) {
        alert("Could not access camera. Please allow permissions.");
      }
    });

    startTaskButton.addEventListener('click', () => {
      if (!model) {
        alert("Please load your model first!");
        return;
      }
      timeCounter = 0;
      classStats = {};
      overlayElem.textContent = "";
      summaryContainer.style.display = "none";
      startTaskButton.disabled = true;
      stopTaskButton.disabled = true;

      let countdownVal = 5;
      countdownElem.textContent = countdownVal;

      countdownInterval = setInterval(() => {
        countdownVal--;
        if (countdownVal > 0) {
          countdownElem.textContent = countdownVal;
        } else {
          clearInterval(countdownInterval);
          countdownElem.textContent = "";
          stopTaskButton.disabled = false;
          startInference();
        }
      }, 1000);
    });

    stopTaskButton.addEventListener('click', () => {
      if (inferenceInterval) {
        clearInterval(inferenceInterval);
      }
      startTaskButton.disabled = false;
      stopTaskButton.disabled = true;
      overlayElem.textContent = "";
      showSummary();
    });

    function startInference() {
      inferenceInterval = setInterval(() => {
        timeCounter++;
        const inputTensor = tf.randomNormal([1, 14739]);
        const predictions = model.predict(inputTensor);
        predictions.data().then(data => {
          let bestIndex = 0;
          let bestScore = data[0];
          for (let i = 1; i < data.length; i++) {
            if (data[i] > bestScore) {
              bestScore = data[i];
              bestIndex = i;
            }
          }
          const confidence = (bestScore * 100).toFixed(1);
          let className = `Class ${bestIndex}`;
          if (metadata && metadata.labels && metadata.labels[bestIndex]) {
            className = metadata.labels[bestIndex];
          }
          overlayElem.textContent = `Time: ${timeCounter}s | ${className} (${confidence}%)`;
          if (!classStats[className]) {
            classStats[className] = { seconds: 0, sumConfidence: 0, count: 0 };
          }
          classStats[className].seconds++;
          classStats[className].sumConfidence += parseFloat(confidence);
          classStats[className].count++;
        });
        predictions.dispose();
        inputTensor.dispose();
      }, 1000);
    }

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

    async function loadModelAndWeights(files) {
      let modelFile = null, weightsFile = null, metaFile = null;
      for (const f of files) {
        const name = f.name.toLowerCase();
        if (name.endsWith("model.json")) modelFile = f;
        else if (name.endsWith(".bin")) weightsFile = f;
        else if (name.endsWith("metadata.json")) metaFile = f;
      }
      if (!modelFile || !weightsFile) throw new Error("model.json or weights.bin missing!");

      if (metaFile) metadata = await parseMetadataFile(metaFile);
      const modelAndWeights = [modelFile, weightsFile];
      try {
        const gModel = await tf.loadGraphModel(tf.io.browserFiles(modelAndWeights));
        return gModel;
      } catch (gErr) {
        try {
          const lModel = await tf.loadLayersModel(tf.io.browserFiles(modelAndWeights));
          return lModel;
        } catch (lErr) {
          throw lErr;
        }
      }
    }

    function parseMetadataFile(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = evt => {
          try {
            const json = JSON.parse(evt.target.result);
            resolve(json);
          } catch (err) {
            reject(err);
          }
        };
        reader.onerror = err => reject(err);
        reader.readAsText(file);
      });
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
