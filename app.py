import os
from flask import Flask, render_template_string

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Pose Duration Tracker with Mr. Lee</title>
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
        /* Adjusted canvas size for better user experience */
        #canvas {
            border: 2px solid #ccc;
            border-radius: 8px;
            width: 640px;
            height: 480px;
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
        #bar-chart-container {
            margin-top: 20px;
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
        @media (max-width: 700px) {
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
    <!-- Include Chart.js from CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
                <!-- Best Match Display Removed -->
            </div>
        </div>
        <div id="bar-chart-container">
            <canvas id="bar-chart"></canvas>
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
        let classDurations = {}; // Object to track durations per class
        let barChart; // Chart.js instance
        let barChartInitialized = false; // Flag to initialize chart once

        // DOM Elements
        const modelUrlInput = document.getElementById('model-url');
        const checkModelButton = document.getElementById('check-model-button');
        const feedback = document.getElementById('feedback');
        const taskSection = document.getElementById('task-section');
        const canvas = document.getElementById('canvas');
        const ctxCanvas = canvas.getContext('2d');
        const overlay = document.getElementById('overlay');
        const taskTimer = document.getElementById('task-timer');
        const startTaskButton = document.getElementById('start-task-button');
        const endTaskButton = document.getElementById('end-task-button');
        const summarySection = document.getElementById('summary-section');
        const summaryContent = document.getElementById('summary-content');
        const restartButton = document.getElementById('restart-button');
        const barChartCanvas = document.getElementById('bar-chart').getContext('2d');

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
            const width = 640; // Standard webcam width
            const height = 480; // Standard webcam height
            const flip = true;
            webcam = new tmPose.Webcam(width, height, flip);
            try {
                console.log("Setting up webcam...");
                await webcam.setup(); // Request webcam access
                await webcam.play();
                console.log("Webcam started.");

                // Dynamically set canvas dimensions to match webcam video
                canvas.width = webcam.canvas.width;
                canvas.height = webcam.canvas.height;

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

        // Function to initialize the bar chart
        function initializeBarChart(labels, initialData) {
            barChart = new Chart(barChartCanvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Pose Probability (%)',
                        data: initialData,
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }

        // Function to update the bar chart
        function updateBarChart(labels, data) {
            if (!barChartInitialized) {
                initializeBarChart(labels, data);
                barChartInitialized = true;
            } else {
                barChart.data.labels = labels;
                barChart.data.datasets[0].data = data;
                barChart.update();
            }
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

                // Prepare data for bar chart
                const labels = prediction.map(pred => pred.className);
                const data = prediction.map(pred => (pred.probability * 100).toFixed(2));

                // Update the bar chart with current probabilities
                updateBarChart(labels, data);

                // Track durations for classes exceeding 80% similarity
                // Find all classes with probability >= 0.8
                const highConfidencePredictions = prediction.filter(pred => pred.probability >= 0.8);

                if (highConfidencePredictions.length > 0) {
                    // Select the prediction with the highest probability
                    const topHighConfidence = highConfidencePredictions.reduce((max, pred) => pred.probability > max.probability ? pred : max, highConfidencePredictions[0]);

                    // Increment duration for the top high confidence class
                    if (classDurations.hasOwnProperty(topHighConfidence.className)) {
                        classDurations[topHighConfidence.className] += 0.1; // Assuming loop runs every 100ms
                    } else {
                        // Initialize if not present
                        classDurations[topHighConfidence.className] = 0.1;
                    }

                    console.log(`Class "${topHighConfidence.className}" exceeded 80% similarity. Total duration: ${classDurations[topHighConfidence.className].toFixed(2)}s`);
                }

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

            // Reset class durations
            for (let className in classDurations) {
                if (classDurations.hasOwnProperty(className)) {
                    classDurations[className] = 0;
                }
            }

            // Initialize bar chart with zero probabilities
            const initialLabels = Object.keys(classDurations);
            const initialData = initialLabels.map(() => 0);
            updateBarChart(initialLabels, initialData);

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

            // Prepare summary data: Show how long each class was over 80% similarity
            const summaryData = Object.entries(classDurations)
                .map(([className, duration]) => ({ className, duration: duration.toFixed(2) }));

            // Sort classes by duration in descending order
            summaryData.sort((a, b) => b.duration - a.duration);

            // Generate summary HTML
            let summaryHTML = `
                <p><strong>Total Task Time:</strong> ${totalTime} seconds</p>
                <p><strong>Duration Each Class Had >80% Similarity:</strong></p>
                <table>
                    <thead>
                        <tr>
                            <th>Class Name</th>
                            <th>Total Duration (s)</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            summaryData.forEach(item => {
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
            taskSection.classList.add('hidden');
            startTaskButton.classList.remove('hidden');
            endTaskButton.classList.add('hidden');
            taskTimer.textContent = "Time: 0.00s";

            // Reset bar chart
            if (barChart) {
                barChart.destroy();
                barChartInitialized = false;
            }

            // Reset class durations
            for (let className in classDurations) {
                if (classDurations.hasOwnProperty(className)) {
                    classDurations[className] = 0;
                }
            }

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
