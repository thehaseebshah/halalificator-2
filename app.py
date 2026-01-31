"""
Halalificator Web API
Flask-based backend for video processing
"""
import os
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename

# Import processing functions
from process_video import halalify

app = Flask(__name__, static_folder='static', static_url_path='')

# Configuration
UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs')
ALLOWED_EXTENSIONS = {'mp4', 'mkv', 'avi', 'mov', 'webm', 'm4v'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Job tracking
jobs = {}  # job_id -> {status, progress, message, output_path, error}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_job(job_id, input_path, options):
    """Background processing task."""
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['message'] = 'Starting video analysis...'
        
        output_filename = f"halalified_{job_id}.mp4"
        output_path = OUTPUT_FOLDER / output_filename
        
        # Run the halalification pipeline
        halalify(
            str(input_path),
            str(output_path),
            conf=options.get('conf', 0.25),
            sensitivity=options.get('sensitivity', 0.15),
            mode=options.get('mode', 'clip'),
            start=options.get('start', 0),
            duration=options.get('duration'),
            debug=options.get('debug', False)
        )
        
        jobs[job_id]['status'] = 'complete'
        jobs[job_id]['message'] = 'Processing complete!'
        jobs[job_id]['output_path'] = str(output_path)
        jobs[job_id]['progress'] = 100
        
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = str(e)
        jobs[job_id]['error'] = str(e)
    finally:
        # Cleanup input file
        try:
            if input_path.exists():
                input_path.unlink()
        except:
            pass


@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle video file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    input_path = UPLOAD_FOLDER / f"{job_id}_{filename}"
    file.save(str(input_path))
    
    # Get processing options from form
    options = {
        'mode': request.form.get('mode', 'clip'),
        'sensitivity': float(request.form.get('sensitivity', 0.15)),
        'conf': float(request.form.get('conf', 0.25)),
        'debug': request.form.get('debug', 'false').lower() == 'true'
    }
    
    # Duration handling
    duration = request.form.get('duration')
    if duration and duration != 'full':
        try:
            options['duration'] = float(duration)
        except ValueError:
            pass
    
    # Initialize job
    jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Queued for processing...',
        'output_path': None,
        'error': None,
        'filename': filename
    }
    
    # Start processing in background thread
    thread = threading.Thread(target=process_job, args=(job_id, input_path, options))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': 'Upload successful, processing started',
        'filename': filename
    })


@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Get job status."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'error': job.get('error'),
        'filename': job.get('filename')
    })


@app.route('/api/download/<job_id>')
def download_file(job_id):
    """Download processed video."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'complete':
        return jsonify({'error': 'Processing not complete'}), 400
    
    output_path = Path(job['output_path'])
    if not output_path.exists():
        return jsonify({'error': 'Output file not found'}), 404
    
    # Generate download filename
    original_name = job.get('filename', 'video.mp4')
    download_name = f"halalified_{original_name}"
    
    return send_file(
        str(output_path),
        as_attachment=True,
        download_name=download_name,
        mimetype='video/mp4'
    )


@app.route('/api/preview/<job_id>')
def preview_video(job_id):
    """Stream video for preview."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'complete':
        return jsonify({'error': 'Processing not complete'}), 400
    
    output_path = Path(job['output_path'])
    if not output_path.exists():
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(str(output_path), mimetype='video/mp4')


if __name__ == '__main__':
    print("=" * 50)
    print("🎥 Halalificator Web Server")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
