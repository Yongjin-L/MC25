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
    <subtitle>KIN217 for MC25</subtitle>
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
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
            gap: 20px; /* Adds space between webcam and chart */
        }
        /* Adjusted canvas size for better user experience */
        #webcam-canvas {
            border: 2px solid #ccc;
            border-radius: 8px;
            width: 320px;
            height: 240px;
        }
        #overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            color: #fff;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 5px;
            font-size: 1em;
            text-align: left;
            width: 180px;
        }
        #feedback-message {
            margin-top: 10px;
            font-size: 1.5em;
            font-weight: bold;
        }
        #countdown {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: red;
            font-size: 3em;
            background: rgba(255, 255, 255, 0.7);
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        #bar-chart-container {
            margin-top: 20px;
            width: 160px; /* Made narrower */
            height: 240px;
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
        #summary-graph-container {
            margin-top: 20px;
            width: 100%;
            max-width: 600px;
            height: 300px;
            margin-left: auto;
            margin-right: auto;
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
            #canvas-container {
                flex-direction: column;
            }
            #webcam-canvas, #bar-chart-container, #summary-graph-container {
                width: 100%;
                height: auto;
            }
            #overlay {
                width: 150px;
                font-size: 1em;
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
    <h1>Pose Duration Tracker with Mr. Lee</h1>

    <!-- Model URL Input Section -->
    <div id="model-section">
        <h2>Load Your Teachable Machine Pose Model</h2>
        <input type="text" id="model-url" placeholder="Enter model URL e.g., https://teachablemachine.withgoogle.com/models/j3EtRGf_g/">
        <br>
        <button type="button" id="check-model-button">Check Model URL</button>
        <button type="button" id="test-webcam-button">Test Webcam</button>
        <button type="button" id="stop-test-webcam-button" class="hidden">Stop Test</button>
        <div id="feedback"></div>
    </div>

    <!-- Task Management Section -->
    <div id="task-section" class="hidden">
        <h2>Pose Matching Task</h2>
        <div id="canvas-container">
            <canvas id="webcam-canvas"></canvas>
            <canvas id="bar-chart"></canvas>
            <div id="overlay">
                <div id="task-timer">Time: 0.00s</div>
                <div id="current-class">Class: N/A</div>
                <div id="current-probability">Probability: 0%</div>
            </div>
            <div id="countdown">5</div>
        </div>
        <div id="feedback-message"></div>
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
        <div id="summary-graph-container">
            <canvas id="summary-graph"></canvas>
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
        let barChart; // Real-time Chart.js instance
        let barChartInitialized = false; // Flag to initialize chart once
        let summaryChart; // Summary Chart.js instance
        let summaryChartInitialized = false; // Flag for summary chart

        // DOM Elements
        const modelUrlInput = document.getElementById('model-url');
        const checkModelButton = document.getElementById('check-model-button');
        const testWebcamButton = document.getElementById('test-webcam-button');
        const stopTestWebcamButton = document.getElementById('stop-test-webcam-button');
        const feedback = document.getElementById('feedback');
        const taskSection = document.getElementById('task-section');
        const webcamCanvas = document.getElementById('webcam-canvas');
        const ctxWebcamCanvas = webcamCanvas.getContext('2d');
        const barChartCanvas = document.getElementById('bar-chart').getContext('2d');
        const overlay = document.getElementById('overlay');
        const taskTimer = document.getElementById('task-timer');
        const currentClass = document.getElementById('current-class');
        const currentProbability = document.getElementById('current-probability');
        const countdownElement = document.getElementById('countdown');
        const feedbackMessage = document.getElementById('feedback-message');
        const startTaskButton = document.getElementById('start-task-button');
        const endTaskButton = document.getElementById('end-task-button');
        const summarySection = document.getElementById('summary-section');
        const summaryContent = document.getElementById('summary-content');
        const summaryGraphCanvas = document.getElementById('summary-graph').getContext('2d');
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

        // Function to initialize the main webcam
        async function setupWebcam() {
            const width = 320; // Standard webcam width for task
            const height = 240; // Standard webcam height for task
            const flip = true;
            webcam = new tmPose.Webcam(width, height, flip);
            try {
                console.log("Setting up main webcam...");
                await webcam.setup(); // Request webcam access
                await webcam.play();
                console.log("Main webcam started.");

                // Set canvas dimensions
                webcamCanvas.width = webcam.canvas.width;
                webcamCanvas.height = webcam.canvas.height;

                window.requestAnimationFrame(loop);
                return true;
            } catch (error) {
                console.error("Error accessing main webcam:", error);
                feedback.textContent = `Error accessing main webcam: ${error.message}`;
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

        // Function to initialize the real-time bar chart
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

        // Function to update the real-time bar chart
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

        // Function to initialize the summary bar chart
        function initializeSummaryChart(labels, data) {
            summaryChart = new Chart(summaryGraphCanvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Total Duration (s)',
                        data: data,
                        backgroundColor: 'rgba(75, 192, 192, 0.7)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            enabled: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        // Function to update the summary bar chart
        function updateSummaryChart(labels, data) {
            if (!summaryChartInitialized) {
                initializeSummaryChart(labels, data);
                summaryChartInitialized = true;
            } else {
                summaryChart.data.labels = labels;
                summaryChart.data.datasets[0].data = data;
                summaryChart.update();
            }
        }

        // Function to start the 5-second countdown
        function startCountdown(duration, display, callback) {
            let timer = duration, seconds;
            display.style.display = 'block';
            display.textContent = timer;
            const countdownInterval = setInterval(() => {
                timer--;
                if (timer >= 0) {
                    display.textContent = timer;
                }
                if (timer < 0) {
                    clearInterval(countdownInterval);
                    display.style.display = 'none';
                    callback();
                }
            }, 1000);
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

                // Prepare data for real-time bar chart
                const labels = prediction.map(pred => pred.className);
                const data = prediction.map(pred => (pred.probability * 100).toFixed(2));

                // Update the real-time bar chart with current probabilities
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

                    // Update overlay with current class and probability
                    currentClass.textContent = `Class: ${topHighConfidence.className}`;
                    currentProbability.textContent = `Probability: ${(topHighConfidence.probability * 100).toFixed(2)}%`;

                    // Provide feedback message based on probability
                    if (topHighConfidence.probability >= 0.8) {
                        feedbackMessage.textContent = "Great Pose!";
                        feedbackMessage.style.color = "green";
                    } else {
                        feedbackMessage.textContent = "Adjust Your Pose.";
                        feedbackMessage.style.color = "red";
                    }

                    console.log(`Class "${topHighConfidence.className}" exceeded 80% similarity. Total duration: ${classDurations[topHighConfidence.className].toFixed(2)}s`);
                } else {
                    // If no class exceeds 80%, reset feedback
                    currentClass.textContent = `Class: N/A`;
                    currentProbability.textContent = `Probability: 0%`;
                    feedbackMessage.textContent = "No Pose Detected.";
                    feedbackMessage.style.color = "orange";
                }

                // Draw the pose
                drawPose(pose, ctxWebcamCanvas);
            } catch (error) {
                console.error("Error during prediction:", error);
                feedback.textContent = `Error during prediction: ${error.message}`;
                feedback.className = "error";
            }
        }

        // Function to draw pose keypoints and skeleton
        function drawPose(pose, ctx) {
            if (webcam.canvas) {
                ctx.drawImage(webcam.canvas, 0, 0, webcamCanvas.width, webcamCanvas.height);
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
                testWebcamButton.disabled = false;
            } else {
                feedback.textContent = "Failed to load model. Please check the URL.";
                feedback.className = "error";
                taskSection.classList.add('hidden');
                startTaskButton.disabled = true;
                testWebcamButton.disabled = true;
            }
        });

        // Event Listener for Test Webcam Button
        testWebcamButton.addEventListener('click', async () => {
            const testSuccess = await setupWebcam();
            if (testSuccess) {
                testWebcamButton.classList.add('hidden');
                stopTestWebcamButton.classList.remove('hidden');
                feedback.textContent = "Webcam test running...";
                feedback.className = "success";
            }
        });

        // Event Listener for Stop Test Webcam Button
        stopTestWebcamButton.addEventListener('click', () => {
            // Stop the webcam
            if (webcam) {
                webcam.stop();
                webcam = null;
                console.log("Webcam stopped.");
            }

            stopTestWebcamButton.classList.add('hidden');
            testWebcamButton.classList.remove('hidden');
            feedback.textContent = "Webcam test stopped.";
            feedback.className = "success";
        });

        // Event Listener for Start Task Button
        startTaskButton.addEventListener('click', async () => {
            // Hide feedback and model section
            document.getElementById('model-section').classList.add('hidden');

            // Start the 5-second countdown
            startCountdown(5, countdownElement, async () => {
                // Initialize webcam for the task
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

                // Initialize real-time bar chart with zero probabilities
                const initialLabels = Object.keys(classDurations);
                const initialData = initialLabels.map(() => 0);
                updateBarChart(initialLabels, initialData);

                // Reset feedback message
                feedbackMessage.textContent = "";

                console.log("Task started.");
            });
        });

        // Event Listener for End Task Button
        endTaskButton.addEventListener('click', () => {
            // Stop the webcam
            if (webcam) {
                webcam.stop();
                webcam = null;
                console.log("Main webcam stopped.");
            }

            // Stop the timer
            clearInterval(taskTimerInterval);
            const totalTime = ((performance.now() - taskStartTime) / 1000).toFixed(2);

            // Prepare summary data: Show total duration of the task
            let summaryHTML = `
                <p><strong>Total Duration:</strong> ${totalTime} seconds</p>
            `;

            summaryContent.innerHTML = summaryHTML;

            // Prepare data for summary graph
            const summaryLabels = Object.keys(classDurations);
            const summaryData = Object.values(classDurations).map(duration => duration.toFixed(2));

            // Generate summary graph
            updateSummaryChart(summaryLabels, summaryData);

            summarySection.style.display = 'block';

            console.log("Task ended. Summary generated.");
            console.log("Total Time:", totalTime);
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
            feedbackMessage.textContent = "";

            // Reset real-time bar chart
            if (barChart) {
                barChart.destroy();
                barChartInitialized = false;
            }

            // Reset summary chart
            if (summaryChart) {
                summaryChart.destroy();
                summaryChartInitialized = false;
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
            testWebcamButton.disabled = true;
            stopTestWebcamButton.classList.add('hidden');

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
