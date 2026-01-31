/**
 * Halalificator Web App
 * Client-side JavaScript for video upload and processing
 */

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const clearFileBtn = document.getElementById('clear-file');
const processBtn = document.getElementById('process-btn');
const sensitivityInput = document.getElementById('sensitivity');
const sensitivityValue = document.getElementById('sensitivity-value');

// Sections
const uploadSection = document.getElementById('upload-section');
const processingSection = document.getElementById('processing-section');
const resultSection = document.getElementById('result-section');
const errorSection = document.getElementById('error-section');

// Processing elements
const processingFilename = document.getElementById('processing-filename');
const progressPercent = document.getElementById('progress-percent');
const progressStatus = document.getElementById('progress-status');
const progressMessage = document.getElementById('progress-message');
const cancelBtn = document.getElementById('cancel-btn');

// Result elements
const resultVideo = document.getElementById('result-video');
const downloadBtn = document.getElementById('download-btn');
const newVideoBtn = document.getElementById('new-video-btn');

// Error elements
const errorMessage = document.getElementById('error-message');
const retryBtn = document.getElementById('retry-btn');

// State
let selectedFile = null;
let currentJobId = null;
let pollingInterval = null;

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showSection(section) {
    [uploadSection, processingSection, resultSection, errorSection].forEach(s => {
        s.classList.add('hidden');
    });
    section.classList.remove('hidden');
}

function setProgress(percent) {
    const circle = document.querySelector('.progress-ring-bar');
    const circumference = 339.292; // 2 * π * 54
    const offset = circumference - (percent / 100) * circumference;
    circle.style.strokeDashoffset = offset;
    progressPercent.textContent = `${Math.round(percent)}%`;
}

function setStep(stepName, status) {
    const steps = ['upload', 'analyze', 'blur', 'audio', 'complete'];
    steps.forEach((step, index) => {
        const stepEl = document.getElementById(`step-${step}`);
        const statusEl = stepEl.querySelector('.step-status');

        if (step === stepName) {
            stepEl.classList.remove('complete');
            stepEl.classList.add('active');
            statusEl.textContent = '●';
        } else if (steps.indexOf(step) < steps.indexOf(stepName)) {
            stepEl.classList.remove('active');
            stepEl.classList.add('complete');
            statusEl.textContent = '✓';
        } else {
            stepEl.classList.remove('active', 'complete');
            statusEl.textContent = '○';
        }
    });
}

// Drag and Drop Handlers
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['video/mp4', 'video/webm', 'video/x-matroska', 'video/avi', 'video/quicktime'];
    const validExtensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'];
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(fileExt)) {
        alert('Please select a valid video file (MP4, MKV, AVI, MOV, WebM)');
        return;
    }

    // Validate file size (500MB max)
    if (file.size > 500 * 1024 * 1024) {
        alert('File size exceeds 500MB limit');
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.classList.remove('hidden');
    processBtn.disabled = false;
}

clearFileBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    processBtn.disabled = true;
});

// Sensitivity slider
sensitivityInput.addEventListener('input', (e) => {
    sensitivityValue.textContent = e.target.value;
});

// Process Button
processBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('mode', document.getElementById('mode').value);
    formData.append('sensitivity', sensitivityInput.value);
    formData.append('duration', document.getElementById('duration').value);
    formData.append('debug', document.getElementById('debug').checked);

    // Switch to processing view
    processingFilename.textContent = selectedFile.name;
    showSection(processingSection);
    setProgress(0);
    setStep('upload', 'active');
    progressStatus.textContent = 'Uploading...';
    progressMessage.textContent = 'Please wait while your video is uploaded';

    try {
        // Upload file
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Upload failed');
        }

        const result = await response.json();
        currentJobId = result.job_id;

        // Start polling for status
        setProgress(10);
        setStep('analyze', 'active');
        progressStatus.textContent = 'Processing...';
        progressMessage.textContent = 'Analyzing video content';

        startPolling();

    } catch (error) {
        showError(error.message);
    }
});

function startPolling() {
    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${currentJobId}`);
            const status = await response.json();

            updateProgress(status);

            if (status.status === 'complete') {
                stopPolling();
                showResult();
            } else if (status.status === 'error') {
                stopPolling();
                showError(status.error || 'Processing failed');
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 1000);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

function updateProgress(status) {
    const message = status.message || '';
    progressMessage.textContent = message;

    // Parse progress from message
    if (message.includes('Analyzing')) {
        setProgress(20);
        setStep('analyze', 'active');
        progressStatus.textContent = 'Analyzing Tracks...';

        // Try to extract frame progress
        const match = message.match(/(\d+)\/(\d+)/);
        if (match) {
            const current = parseInt(match[1]);
            const total = parseInt(match[2]);
            const pct = 20 + (current / total) * 30;
            setProgress(Math.min(50, pct));
        }
    } else if (message.includes('Rendering')) {
        setProgress(55);
        setStep('blur', 'active');
        progressStatus.textContent = 'Rendering Video...';

        const match = message.match(/(\d+)\/(\d+)/);
        if (match) {
            const current = parseInt(match[1]);
            const total = parseInt(match[2]);
            const pct = 55 + (current / total) * 20;
            setProgress(Math.min(75, pct));
        }
    } else if (message.includes('Audio') || message.includes('Demucs') || message.includes('vocals')) {
        setProgress(80);
        setStep('audio', 'active');
        progressStatus.textContent = 'Processing Audio...';
    } else if (message.includes('Combining') || message.includes('merge')) {
        setProgress(90);
        setStep('audio', 'active');
        progressStatus.textContent = 'Finalizing...';
    } else if (message.includes('complete')) {
        setProgress(100);
        setStep('complete', 'active');
        progressStatus.textContent = 'Complete!';
    }
}

function showResult() {
    setProgress(100);
    setStep('complete', 'active');

    // Set video source
    resultVideo.src = `/api/preview/${currentJobId}`;
    resultVideo.load();

    // Show result section
    showSection(resultSection);
}

function showError(message) {
    errorMessage.textContent = message;
    showSection(errorSection);
}

// Download button
downloadBtn.addEventListener('click', () => {
    if (currentJobId) {
        window.location.href = `/api/download/${currentJobId}`;
    }
});

// New video button
newVideoBtn.addEventListener('click', resetApp);
retryBtn.addEventListener('click', resetApp);

function resetApp() {
    // Reset state
    selectedFile = null;
    currentJobId = null;
    stopPolling();

    // Reset UI
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    processBtn.disabled = true;
    resultVideo.src = '';

    // Reset steps
    ['upload', 'analyze', 'blur', 'audio', 'complete'].forEach(step => {
        const stepEl = document.getElementById(`step-${step}`);
        stepEl.classList.remove('active', 'complete');
        stepEl.querySelector('.step-status').textContent = '○';
    });

    // Show upload section
    showSection(uploadSection);
}

// Cancel button
cancelBtn.addEventListener('click', () => {
    stopPolling();
    resetApp();
});

// Add SVG gradient for progress ring (needs to be in the DOM)
document.addEventListener('DOMContentLoaded', () => {
    const svg = document.querySelector('.progress-ring-svg');
    if (svg) {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        defs.innerHTML = `
            <linearGradient id="progress-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#6366f1"/>
                <stop offset="50%" style="stop-color:#8b5cf6"/>
                <stop offset="100%" style="stop-color:#a855f7"/>
            </linearGradient>
        `;
        svg.insertBefore(defs, svg.firstChild);
    }
});
