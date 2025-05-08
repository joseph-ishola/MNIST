document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements - Drawing
    const canvas = document.getElementById('drawing-board');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clear-button');
    const predictButton = document.getElementById('predict-button');
    const penThickness = document.getElementById('pen-thickness');
    const thicknessValue = document.getElementById('thickness-value');
    
    // Get DOM elements - Results
    const resultContainer = document.getElementById('result-container');
    const predictedDigit = document.getElementById('predicted-digit');
    const topPredictionsContainer = document.getElementById('top-predictions-container');
    const loadingIndicator = document.getElementById('loading');
    const processedImage = document.getElementById('processed-image');
    const resultModelName = document.getElementById('result-model-name');
    const loadingModelName = document.getElementById('loading-model-name');
    
    // Get DOM elements - Model Selection
    const modelSelector = document.getElementById('model-selector');
    const modelDescription = document.getElementById('model-description');
    
    // Get DOM elements - Comparison
    const compareButton = document.getElementById('compare-button');
    const comparisonLoading = document.getElementById('comparison-loading');
    const comparisonResults = document.getElementById('comparison-results');
    const comparisonContainer = document.getElementById('comparison-container');
    
    // Drawing state
    let isDrawing = false;
    
    // Set up drawing parameters
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';
    
    // Initialize canvas with black background
    initCanvas();
    
    // Event listeners - Drawing
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
    
    // Event listeners - Model Selection
    modelSelector.addEventListener('change', updateModelDescription);
    
    // Event listeners - Comparison
    compareButton.addEventListener('click', compareAllModels);
    
    // Enable auto-prediction with increased delay
    canvas.addEventListener('mouseup', debounce(predictDigit, 400));
    canvas.addEventListener('touchend', debounce(predictDigit, 400));
    
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
        comparisonResults.style.display = 'none';
    }
    
    /**
     * Send the drawing to server for prediction with the selected model
     */
    function predictDigit() {
        // Check if canvas is empty (all black)
        if (!hasDrawing()) {
            alert('Please draw a digit first!');
            return;
        }
        
        // Get the selected model
        const selectedModel = modelSelector.value;
        loadingModelName.textContent = getModelDisplayName(selectedModel);
        
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
                image: imageData64,
                model: selectedModel
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
        // Check if there was an error
        if (data.error) {
            alert('Error: ' + data.error);
            loadingIndicator.style.display = 'none';
            return;
        }
        
        // Update model name in result
        resultModelName.textContent = data.model_display_name || data.model;
        
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
    
    /**
     * Compare the drawing with all available models
     */
    function compareAllModels() {
        // Check if canvas is empty (all black)
        if (!hasDrawing()) {
            alert('Please draw a digit first!');
            return;
        }
        
        // Show loading indicator
        comparisonLoading.style.display = 'block';
        comparisonResults.style.display = 'none';
        comparisonContainer.innerHTML = '';
        
        // Convert canvas to base64 image
        const imageData64 = canvas.toDataURL('image/png');
        
        // Get list of models
        fetch('/list_models')
            .then(response => response.json())
            .then(models => {
                // Create an array of promises for each model prediction
                const predictionPromises = Object.keys(models).map(modelName => {
                    return fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            image: imageData64,
                            model: modelName
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        // Add model info to the result
                        data.modelName = modelName;
                        data.displayName = models[modelName].display_name;
                        return data;
                    });
                });
                
                // Wait for all predictions to complete
                return Promise.all(predictionPromises);
            })
            .then(allResults => {
                // Sort results by confidence (highest first)
                allResults.sort((a, b) => {
                    const aConf = a.top_3[0].confidence;
                    const bConf = b.top_3[0].confidence;
                    return bConf - aConf;
                });
                
                // Display all results
                displayModelComparison(allResults);
            })
            .catch(error => {
                console.error('Error in model comparison:', error);
                comparisonLoading.style.display = 'none';
                alert('Error comparing models. Please try again.');
            });
    }
    
    /**
     * Display model comparison results
     */
    function displayModelComparison(results) {
        comparisonContainer.innerHTML = '';
        
        // Create a card for each model result
        results.forEach((result, index) => {
            const isConsensus = index > 0 && result.digit === results[0].digit;
            const confidence = (result.top_3[0].confidence * 100).toFixed(2);
            
            const cardHtml = `
                <div class="col-md-4 mb-4">
                    <div class="card ${isConsensus ? 'border-success' : ''}">
                        <div class="card-header ${isConsensus ? 'bg-success text-white' : 'bg-primary text-white'}">
                            <h5 class="mb-0">${result.displayName || result.modelName}</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-4 text-center">
                                    <div class="comparison-digit">${result.digit}</div>
                                    <div class="comparison-confidence">${confidence}%</div>
                                </div>
                                <div class="col-8">
                                    <div class="small">
                                        <strong>Top Predictions:</strong>
                                        ${result.top_3.map(pred => {
                                            const predConf = (pred.confidence * 100).toFixed(2);
                                            return `
                                                <div class="mb-1">
                                                    <div class="d-flex justify-content-between">
                                                        <span>Digit ${pred.digit}</span>
                                                        <span>${predConf}%</span>
                                                    </div>
                                                    <div class="progress" style="height: 6px;">
                                                        <div class="progress-bar" role="progressbar" style="width: ${predConf}%;" 
                                                            aria-valuenow="${predConf}" aria-valuemin="0" aria-valuemax="100"></div>
                                                    </div>
                                                </div>
                                            `;
                                        }).join('')}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            comparisonContainer.innerHTML += cardHtml;
        });
        
        // Show comparison results and hide loading
        comparisonLoading.style.display = 'none';
        comparisonResults.style.display = 'block';
        
        // Add summary analysis
        const consensusDigit = getMostCommonDigit(results);
        const consensusCount = results.filter(r => r.digit === consensusDigit).length;
        const consensusPercent = ((consensusCount / results.length) * 100).toFixed(0);
        
        const summaryHtml = `
            <div class="col-12 mb-4">
                <div class="alert ${consensusPercent > 50 ? 'alert-success' : 'alert-warning'}">
                    <h5>Consensus Analysis</h5>
                    <p>
                        ${consensusCount} out of ${results.length} models (${consensusPercent}%) 
                        identified this as digit <strong>${consensusDigit}</strong>.
                        ${consensusPercent > 80 ? 'Strong consensus among models!' : 
                          consensusPercent > 50 ? 'Moderate consensus among models.' : 
                          'Models disagree on the classification.'}
                    </p>
                </div>
            </div>
        `;
        
        comparisonContainer.insertAdjacentHTML('afterbegin', summaryHtml);
    }
    
    /**
     * Get most common digit prediction from results
     */
    function getMostCommonDigit(results) {
        const digitCounts = {};
        let maxCount = 0;
        let mostCommonDigit = null;
        
        results.forEach(result => {
            const digit = result.digit;
            digitCounts[digit] = (digitCounts[digit] || 0) + 1;
            
            if (digitCounts[digit] > maxCount) {
                maxCount = digitCounts[digit];
                mostCommonDigit = digit;
            }
        });
        
        return mostCommonDigit;
    }
    
    /**
     * Update the model description when a new model is selected
     */
    function updateModelDescription() {
        const selectedModel = modelSelector.value;
        
        // Get model info from the server
        fetch('/list_models')
            .then(response => response.json())
            .then(models => {
                if (models[selectedModel] && models[selectedModel].description) {
                    modelDescription.textContent = models[selectedModel].description;
                } else {
                    modelDescription.textContent = `Model: ${selectedModel}`;
                }
            })
            .catch(error => {
                console.error('Error fetching model description:', error);
                modelDescription.textContent = `Model: ${selectedModel}`;
            });
    }
    
    /**
     * Get display name for a model
     */
    function getModelDisplayName(modelName) {
        const option = Array.from(modelSelector.options).find(opt => opt.value === modelName);
        return option ? option.text : modelName;
    }
});