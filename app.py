import os
from flask import Flask, render_template_string

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Custom Pose Duration Tracker</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f7f7f7;
            margin: 0;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        #model-section, #task-section, #summary-section {
            margin: 20px auto;
            padding: 20px;
            width: 90%;
            max-width: 800px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        #model-section input {
            width: 80%;
            padding: 10px;
            font-size: 1em;
            margin-bottom: 10px;
        }
        #model-section button, #task-section button, #summary-section button {
            padding: 10px 20px;
            font-size: 1em;
            margin: 5px;
            cursor: pointer;
        }
        #feedback {
            margin-top: 10px;
            font-weight: bold;
        }
        #canvas-container {
            position: relative;
            display: inline-block;
            margin-top: 20px;
        }
        #canvas {
            border: 2px solid #ccc;
            border-radius: 8px;
            width: 800px;
            height: 600px;
        }
        #overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            color: #fff;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 5px;
            font-size: 1.2em;
            text-align: left;
        }
        #summary-section {
            display: none;
        }
        #summary-section h2 {
            color: #444;
        }
        #summary-content {
            text-align: left;
            margin-top: 10px;
        }
        .hidden {
            display: none;
        }
        .error {
            color: red;
        }
        .success {
            color: green;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        table, th, td {
            border: 1px solid #ccc;
        }
        th, td {
            padding: 10px;
            text-align: center;
        }
        th {
            background-color: #f0f0f0;
        }
        @media (max-width: 820px) {
            #canvas {
                width: 100%;
                height: auto;
            }
            #model-section input {
                width: 100%;
                margin-bottom: 10px;
            }
            #model-section button, #task-section button, #summary-section button {
                width: 100%;
                margin: 5px 0;
            }
        }
    </style>
</head>
<body>
    <h1>Custom Pose Duration Tracker</h1>

    <!-- Model URL Input Section -->
    <div id="model-section">
        <h2>Load Your Teachable Machine Pose Model</h2>
        <input type="text" id="model-url" placeholder="Enter model URL e.g., https://teachablemachine.withgoogle.com/models/j3EtRGf_g/">
        <br>
        <button type="button" id="check-model-button">Check Model URL</button>
        <div id="feedback"></div>
    </div>

    <!-- Task Management Section -->
    <div id="task-section" class="hidden">
        <h2>Pose Matching Task</h2>
        <div id="canvas-container">
            <canvas id="canvas"></canvas>
            <div id="overlay">
                <div id="task-timer">Time: 0.00s</div>
                <div id="best-match">Best Match: N/A</div>
            </div>
        </div>
        <br>
        <button type="button" id="start-task-button" disabled>Start Task</button>
        <button type="button" id="end-task-button" class="hidden">End Task</button>
    </div>

    <!-- Summary Section -->
    <div id="summary-section">
        <h2>Task Summary</h2>
        <div id="summary-content">
            <!-- Summary details will be populated here -->
        </div>
        <button type="button" id="restart-button">Restart</button>
    </div>

    <!-- TensorFlow.js and Teachable Machine Pose libraries (Using Version 0.8) -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/pose@0.8/dist/teachablemachine-pose.min.js"></script>

    <script type="text/javascript">
        // Initial Variables
        let model, webcam, ctx, maxPredictions;
        let taskTimerInterval;
        let taskStartTime = null;
        let bestMatch = { className: 'N/A', probability: 0 };
        let classDurations = {}; // Object to track durations per class

        // DOM Elements
        const modelUrlInput = document.getElementById('model-url');
        const checkModelButton = document.getElementById('check-model-button');
        const feedback = document.getElementById('feedback');
        const taskSection = document.getElementById('task-section');
        const canvas = document.getElementById('canvas');
        const ctxCanvas = canvas.getContext('2d');
        const overlay = document.getElementById('overlay');
        const taskTimer = document.getElementById('task-timer');
        const bestMatchDisplay = document.getElementById('best-match');
        const startTaskButton = document.getElementById('start-task-button');
        const endTaskButton = document.getElementById('end-task-button');
        const summarySection = document.getElementById('summary-section');
        const summaryContent = document.getElementById('summary-content');
        const restartButton = document.getElementById('restart-button');

        // Function to validate and load the model
        async function loadModel(modelURL) {
            try {
                // Ensure the URL ends with a slash
                if (!modelURL.endsWith('/')) {
                    modelURL += '/';
                }

                console.log(`Loading model from ${modelURL}`);

                const modelJSON = modelURL + "model.json";
                const metadataJSON = modelURL + "metadata.json";

                // Attempt to load the model
                model = await tmPose.load(modelJSON, metadataJSON);
                maxPredictions = model.getTotalClasses();

                console.log(`Model loaded with ${maxPredictions} classes.`);

                // Initialize classDurations
                classDurations = {};
                for (let i = 0; i < maxPredictions; i++) {
                    let className = null;

                    // Attempt to retrieve class names from model.classes
                    if (model.classes && model.classes.length > 0) {
                        className = model.classes[i];
                    }

                    // Fallback: Attempt to retrieve class names from model.labels
                    if (!className && model.labels && model.labels.length > 0) {
                        className = model.labels[i];
                    }

                    // Fallback: If className is still not found, use prediction class names later
                    if (!className) {
                        console.warn(`Class name at index ${i} is undefined. It will be tracked dynamically.`);
                        continue;
                    }

                    classDurations[className] = 0; // Initialize duration to 0
                }

                // If successful, return true
                return true;
            } catch (error) {
                console.error("Error loading model:", error);
                feedback.textContent = `Error loading model: ${error.message}`;
                feedback.className = "error";
                return false;
            }
        }

        // Function to initialize the webcam
        async function setupWebcam() {
            const width = 800;
            const height = 600;
            const flip = true;
            webcam = new tmPose.Webcam(width, height, flip);
            try {
                console.log("Setting up webcam...");
                await webcam.setup(); // Request webcam access
                await webcam.play();
                console.log("Webcam started.");
                window.requestAnimationFrame(loop);
                return true;
            } catch (error) {
                console.error("Error accessing webcam:", error);
                feedback.textContent = `Error accessing webcam: ${error.message}`;
                feedback.className = "error";
                return false;
            }
        }

        // Function to update the timer
        function updateTaskTimer() {
            const currentTime = performance.now();
            const elapsed = ((currentTime - taskStartTime) / 1000).toFixed(2);
            taskTimer.textContent = `Time: ${elapsed}s`;
        }

        // Main prediction loop
        async function loop() {
            if (!webcam) return;

            webcam.update(); // Update the webcam frame
            await predict();
            window.requestAnimationFrame(loop);
        }

        // Prediction function
        async function predict() {
            if (!model || !webcam.canvas) return;

            try {
                const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
                const prediction = await model.predict(posenetOutput);

                // Log predictions for debugging
                console.log("Predictions:", prediction);

                // Determine the best prediction
                let topPrediction = prediction.reduce((max, pred) => pred.probability > max.probability ? pred : max, prediction[0]);

                // Update best match if current prediction is better
                if (topPrediction.probability > bestMatch.probability) {
                    bestMatch = {
                        className: topPrediction.className,
                        probability: (topPrediction.probability * 100).toFixed(2) + "%"
                    };
                    bestMatchDisplay.textContent = `Best Match: ${bestMatch.className} (${bestMatch.probability})`;
                    console.log(`New best match: ${bestMatch.className} with ${bestMatch.probability} confidence.`);
                }

                // Track durations for classes exceeding 80% similarity
                prediction.forEach(pred => {
                    if (pred.probability >= 0.8) {
                        // If className exists in classDurations, increment its duration
                        if (classDurations.hasOwnProperty(pred.className)) {
                            classDurations[pred.className] += 0.1; // Assuming loop runs every 100ms
                            console.log(`Class "${pred.className}" exceeded 80% similarity. Total duration: ${classDurations[pred.className].toFixed(2)}s`);
                        } else {
                            // If className not tracked yet, initialize it
                            classDurations[pred.className] = 0.1;
                            console.log(`Class "${pred.className}" exceeded 80% similarity. Total duration: ${classDurations[pred.className].toFixed(2)}s`);
                        }
                    }
                });

                // Draw the pose
                drawPose(pose, ctxCanvas);
            } catch (error) {
                console.error("Error during prediction:", error);
                feedback.textContent = `Error during prediction: ${error.message}`;
                feedback.className = "error";
            }
        }

        // Function to draw pose keypoints and skeleton
        function drawPose(pose, ctx) {
            if (webcam.canvas) {
                ctx.drawImage(webcam.canvas, 0, 0, canvas.width, canvas.height);
                if (pose) {
                    const minPartConfidence = 0.5;
                    tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
                    tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
                }
            }
        }

        // Event Listener for Check Model Button
        checkModelButton.addEventListener('click', async () => {
            const modelURL = modelUrlInput.value.trim();
            if (!modelURL) {
                feedback.textContent = "Please enter a model URL.";
                feedback.className = "error";
                return;
            }

            feedback.textContent = "Checking model URL...";
            feedback.className = "";

            const isValid = await loadModel(modelURL);

            if (isValid) {
                feedback.textContent = "Model loaded successfully!";
                feedback.className = "success";
                taskSection.classList.remove('hidden');
                startTaskButton.disabled = false;
            } else {
                feedback.textContent = "Failed to load model. Please check the URL.";
                feedback.className = "error";
                taskSection.classList.add('hidden');
                startTaskButton.disabled = true;
            }
        });

        // Event Listener for Start Task Button
        startTaskButton.addEventListener('click', async () => {
            // Hide feedback and model section
            document.getElementById('model-section').classList.add('hidden');

            // Initialize webcam
            const webcamSetupSuccess = await setupWebcam();
            if (!webcamSetupSuccess) {
                return;
            }

            // Show task section elements
            startTaskButton.classList.add('hidden');
            endTaskButton.classList.remove('hidden');

            // Start the timer
            taskStartTime = performance.now();
            taskTimerInterval = setInterval(updateTaskTimer, 100);

            // Reset best match
            bestMatch = { className: 'N/A', probability: 0 };
            bestMatchDisplay.textContent = "Best Match: N/A";

            // Reset class durations
            for (let className in classDurations) {
                if (classDurations.hasOwnProperty(className)) {
                    classDurations[className] = 0;
                }
            }

            console.log("Task started.");
        });

        // Event Listener for End Task Button
        endTaskButton.addEventListener('click', () => {
            // Stop the webcam
            if (webcam) {
                webcam.stop();
                webcam = null;
                console.log("Webcam stopped.");
            }

            // Stop the timer
            clearInterval(taskTimerInterval);
            const totalTime = ((performance.now() - taskStartTime) / 1000).toFixed(2);

            // Prepare summary data: Filter classes with duration >= 0.8s (assuming each increment is 0.1s, 0.8s = 8 increments)
            const filteredClasses = Object.entries(classDurations)
                .filter(([className, duration]) => duration >= 0.8)
                .map(([className, duration]) => ({ className, duration: duration.toFixed(2) }));

            // Sort classes by duration in descending order
            filteredClasses.sort((a, b) => b.duration - a.duration);

            // Generate summary HTML
            let summaryHTML = `
                <p><strong>Total Task Time:</strong> ${totalTime} seconds</p>
            `;

            if (filteredClasses.length > 0) {
                summaryHTML += `
                    <p><strong>Classes with >80% Similarity:</strong></p>
                    <table>
                        <thead>
                            <tr>
                                <th>Class Name</th>
                                <th>Total Duration (s)</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                filteredClasses.forEach(item => {
                    summaryHTML += `
                        <tr>
                            <td>${item.className}</td>
                            <td>${item.duration}</td>
                        </tr>
                    `;
                });
                summaryHTML += `
                        </tbody>
                    </table>
                `;
            } else {
                summaryHTML += `<p>No classes reached 80% similarity during the task.</p>`;
            }

            summaryContent.innerHTML = summaryHTML;
            summarySection.style.display = 'block';

            console.log("Task ended. Summary generated.");
            console.log("Class Durations:", classDurations);

            // Hide task section elements
            taskSection.classList.add('hidden');
        });

        // Event Listener for Restart Button
        restartButton.addEventListener('click', () => {
            // Reset variables
            summarySection.style.display = 'none';
            taskSection.classList.remove('hidden');
            startTaskButton.classList.remove('hidden');
            endTaskButton.classList.add('hidden');
            taskTimer.textContent = "Time: 0.00s";
            bestMatchDisplay.textContent = "Best Match: N/A";

            // Reset model section
            document.getElementById('model-section').classList.remove('hidden');
            modelUrlInput.value = "";
            feedback.textContent = "";
            feedback.className = "";
            startTaskButton.disabled = true;

            console.log("Application restarted.");
        });

        // Initialize the application on page load
        window.addEventListener('load', () => {
            // Initially hide task and summary sections
            taskSection.classList.add('hidden');
            summarySection.style.display = 'none';
            console.log("Application loaded. Awaiting user input.");
        });
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
