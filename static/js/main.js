document.addEventListener('DOMContentLoaded', function () {
    // --- DOM Element References ---
    const fileInput = document.getElementById('fileInput');
    const uploadSection = document.getElementById('uploadSection');
    const processingSection = document.getElementById('processingSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const uploadCard = document.querySelector('.upload-card');

    let originalImageData = null;
    let processingInterval = null;

    // --- Event Listeners ---
    fileInput.addEventListener('change', handleFileSelect);
    uploadCard.addEventListener('click', () => fileInput.click());
    uploadCard.addEventListener('dragover', handleDragOver);
    uploadCard.addEventListener('dragleave', handleDragLeave);
    uploadCard.addEventListener('drop', handleDrop);

    // --- Core Functions ---

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) handleFile(file);
    }

    function handleDragOver(e) {
        e.preventDefault();
        uploadCard.classList.add('drag-over');
    }

    function handleDragLeave() {
        uploadCard.classList.remove('drag-over');
    }

    function handleDrop(e) {
        e.preventDefault();
        uploadCard.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    }

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showError('Please upload a valid image file (JPG, PNG, etc.).');
            return;
        }
        if (file.size > 16 * 1024 * 1024) {
            showError('Image file size cannot exceed 16MB.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            originalImageData = e.target.result;
            document.getElementById('originalImage').src = originalImageData;
            sendImageToServer(file);
        };
        reader.readAsDataURL(file);
    }

    async function sendImageToServer(file) {
        showSection('processingSection');
        animateProcessingSteps();

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/process', { method: 'POST', body: formData });
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'An unknown server error occurred.');
            }
            displayResults(result);
        } catch (error) {
            console.error('Processing failed:', error);
            showError(error.message);
        }
    }

    function displayResults(data) {
        clearInterval(processingInterval);
        showSection('resultsSection');

        document.getElementById('processedImage').src = 'data:image/png;base64,' + data.image;

        const statsEl = document.getElementById('resultsStats');
        statsEl.innerHTML = `<i class="fas fa-search"></i> Found ${data.detections.length} text region${data.detections.length !== 1 ? 's' : ''}`;

        const textItemsEl = document.getElementById('textItems');
        textItemsEl.innerHTML = ''; // Clear previous results

        if (data.detections.length === 0) {
            textItemsEl.innerHTML = `<p class="no-text-found">No text was detected in the image.</p>`;
        } else {
            data.detections.forEach((detection, index) => {
                const itemEl = createTextItem(index + 1, detection);
                textItemsEl.appendChild(itemEl);
            });
        }
    }

    function createTextItem(index, detection) {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'text-item';

        const rawText = detection.raw_text || '(empty)';
        const regenText = detection.regenerated_text;
        const confidence = detection.confidence || 0;
        const wasCorrected = detection.was_corrected;

        // Determine confidence color and icon
        const confPercent = (confidence * 100).toFixed(1);
        let confColor = 'var(--danger)';
        let confIcon = 'fa-times-circle';
        if (confidence > 0.75) {
            confColor = 'var(--success)';
            confIcon = 'fa-check-circle';
        } else if (confidence > 0.4) {
            confColor = 'var(--warning)';
            confIcon = 'fa-exclamation-circle';
        }

        itemDiv.innerHTML = `
            <div class="text-item-header">
                <strong>Region ${index}</strong>
                <span class="confidence-badge" style="color: ${confColor};">
                    <i class="fas ${confIcon}"></i> ${confPercent}% Confidence
                </span>
            </div>
            <div class="text-field">
                <label>RAW OCR OUTPUT</label>
                <div class="text-value raw">${escapeHtml(rawText)}</div>
            </div>
            <div class="text-field">
                <label>${wasCorrected ? '<i class="fas fa-magic"></i> REGENERATED TEXT' : 'FINAL TEXT'}</label>
                <div class="text-value final ${wasCorrected ? 'corrected' : ''}">${escapeHtml(regenText)}</div>
            </div>
        `;
        return itemDiv;
    }

    function animateProcessingSteps() {
        const steps = ['step1', 'step2', 'step3', 'step4'];
        steps.forEach(id => document.getElementById(id).classList.remove('active'));
        let currentStep = 0;
        processingInterval = setInterval(() => {
            if (currentStep < steps.length) {
                document.getElementById(steps[currentStep]).classList.add('active');
                currentStep++;
            } else {
                clearInterval(processingInterval);
            }
        }, 600);
    }

    function showSection(sectionId) {
        ['uploadSection', 'processingSection', 'resultsSection', 'errorSection'].forEach(id => {
            document.getElementById(id).style.display = 'none';
        });
        document.getElementById(sectionId).style.display = 'block';
    }

    function showError(message) {
        clearInterval(processingInterval);
        showSection('errorSection');
        document.getElementById('errorMessage').textContent = message;
    }

    window.resetApp = function() {
        showSection('uploadSection');
        fileInput.value = '';
        originalImageData = null;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});