document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const canvas = document.getElementById('drawing-board');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clear-button');
    const predictButton = document.getElementById('predict-button');
    const resultContainer = document.getElementById('result-container');
    const predictedDigit = document.getElementById('predicted-digit');
    const topPredictionsContainer = document.getElementById('top-predictions-container');
    const loadingIndicator = document.getElementById('loading');
    const processedImage = document.getElementById('processed-image');
    const penThickness = document.getElementById('pen-thickness');
    const thicknessValue = document.getElementById('thickness-value');

    // Drawing state
    let isDrawing = false;

    // Set up drawing parameters
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';

    // Initialize canvas with black background
    initCanvas();

    // Event listeners
    penThickness.addEventListener('input', updatePenThickness);
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    clearButton.addEventListener('click', clearCanvas);
    predictButton.addEventListener('click', predictDigit);

    // Enable auto-prediction
    canvas.addEventListener('mouseup', debounce(predictDigit, 500));
    canvas.addEventListener('touchend', debounce(predictDigit, 500));

    /**
     * Initialize the canvas with a black background
     */
    function initCanvas() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    /**
     * Update pen thickness from slider
     */
    function updatePenThickness() {
        const thickness = parseInt(this.value);
        ctx.lineWidth = thickness;
        thicknessValue.textContent = thickness;
    }

    /**
     * Start drawing when mouse/touch is down
     */
    function startDrawing(e) {
        isDrawing = true;
        draw(e);
    }

    /**
     * Draw on canvas as mouse/touch moves
     */
    function draw(e) {
        if (!isDrawing) return;
        
        const rect = canvas.getBoundingClientRect();
        let x, y;
        
        if (e.type.includes('mouse')) {
            x = e.clientX - rect.left;
            y = e.clientY - rect.top;
        } else {
            // Prevent scrolling when drawing
            e.preventDefault();
            x = e.touches[0].clientX - rect.left;
            y = e.touches[0].clientY - rect.top;
        }
        
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    /**
     * Stop drawing when mouse/touch is released or leaves canvas
     */
    function stopDrawing() {
        isDrawing = false;
        ctx.beginPath();
    }

    /**
     * Handle touch events by converting to mouse events
     */
    function handleTouch(e) {
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mouse' + e.type.replace('touch', ''), {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }

    /**
     * Clear the canvas
     */
    function clearCanvas() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultContainer.style.display = 'none';
    }

    /**
     * Send the drawing to server for prediction
     */
    function predictDigit() {
        // Check if canvas is empty (all black)
        if (!hasDrawing()) {
            alert('Please draw a digit first!');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultContainer.style.display = 'none';
        
        // Convert canvas to base64 image
        const imageData64 = canvas.toDataURL('image/png');
        
        // Send to server for prediction
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData64
            })
        })
        .then(response => response.json())
        .then(handlePredictionResponse)
        .catch(handlePredictionError);
    }

    /**
     * Check if canvas has any drawing (not all black)
     */
    function hasDrawing() {
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        for (let i = 0; i < imageData.length; i += 4) {
            if (imageData[i] > 0 || imageData[i+1] > 0 || imageData[i+2] > 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * Handle successful prediction response
     */
    function handlePredictionResponse(data) {
        // Display the prediction
        predictedDigit.textContent = data.digit;
        
        // Display top 3 predictions with confidence bars
        topPredictionsContainer.innerHTML = '';
        data.top_3.forEach(pred => {
            const confidence = (pred.confidence * 100).toFixed(2);
            const predHtml = `
                <div class="mb-2">
                    <div class="d-flex justify-content-between mb-1">
                        <strong>Digit ${pred.digit}</strong>
                        <span>${confidence}%</span>
                    </div>
                    <div class="progress" style="height: 10px;">
                        <div class="progress-bar" role="progressbar" style="width: ${confidence}%;" 
                            aria-valuenow="${confidence}" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                </div>
            `;
            topPredictionsContainer.innerHTML += predHtml;
        });
        
        // Display the processed image
        processedImage.src = `data:image/png;base64,${data.processed_image}`;
        
        // Hide loading indicator and show results
        loadingIndicator.style.display = 'none';
        resultContainer.style.display = 'block';
    }

    /**
     * Handle prediction errors
     */
    function handlePredictionError(error) {
        console.error('Error:', error);
        loadingIndicator.style.display = 'none';
        alert('Error processing the image. Please try again.');
    }
    
    /**
     * Utility function for debouncing
     */
    function debounce(func, wait) {
        let timeout;
        return function() {
            const context = this;
            const args = arguments;
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(context, args), wait);
        };
    }
});