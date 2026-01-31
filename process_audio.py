import os
import subprocess
import shutil
import argparse
import sys
from pathlib import Path

def find_demucs():
    """Locate the demucs executable."""
    # Check in the same directory as the python executable (e.g. inside .venv/bin)
    candidate = Path(sys.executable).parent / "demucs"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)
    
    # Check system PATH
    path_demucs = shutil.which("demucs")
    if path_demucs:
        return path_demucs
        
    return "demucs" # Hope it's in path if all else fails

def remove_music(input_path, output_path=None, model="htdemucs"):
    """
    Separates vocals from audio using Demucs, effectively removing music.
    Returns the path to the isolated vocals file.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Processing audio input: {input_path}")
    
    demucs_cmd = find_demucs()
    print(f"Using Demucs executable: {demucs_cmd}")

    # Ensure ffmpeg is in path for Demucs
    env = os.environ.copy()
    try:
        from static_ffmpeg import add_paths
        add_paths()
        # static_ffmpeg adds to os.environ["PATH"]
        env["PATH"] = os.environ["PATH"]
        print("Integrated static-ffmpeg into PATH.")
    except ImportError:
        print("static-ffmpeg not found, relying on system PATH for ffmpeg.")

    # Run Demucs
    # --two-stems=vocals: splits into 'vocals' (speech/singing) and 'no_vocals' (music/accompaniment)
    cmd = [
        demucs_cmd,
        "--two-stems=vocals",
        "-n", model,
        str(input_path)
    ]
    
    print(f"Running Demucs with model {model}... (this may take a few minutes)")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Demucs failed with error code {e.returncode}")
    
    # Locate output
    # Demucs outputs to separated/<model>/<track_name>/vocals.wav
    filename_stem = input_path.stem
    
    # Demucs output directory logic
    output_root = Path("separated") / model
    
    # Try exact match first
    candidate_dir = output_root / filename_stem
    vocals_file = candidate_dir / "vocals.wav"
    
    if not vocals_file.exists():
        print(f"Exact output directory not found at {candidate_dir}, searching...")
        found = False
        if output_root.exists():
            for child in output_root.iterdir():
                if child.is_dir():
                    # Heuristic match
                    if child.name == filename_stem or child.name.replace(' ', '_') == filename_stem.replace(' ', '_'):
                         vocals_file = child / "vocals.wav"
                         if vocals_file.exists():
                             candidate_dir = child
                             found = True
                             break
        
        if not found:
             raise RuntimeError(f"Could not locate separated output for {filename_stem}")

    print(f"Separation complete. Vocals isolated at: {vocals_file}")
    
    # Move to desired output path
    if output_path:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(vocals_file), str(output_path))
        target_file = output_path
    else:
        # Default: input_name_vocals.wav
        target_file = input_path.with_name(f"{input_path.stem}_vocals.wav")
        if target_file.exists():
             os.remove(target_file)
        shutil.move(str(vocals_file), str(target_file))

    # Cleanup the specific track folder
    try:
        shutil.rmtree(candidate_dir)
        try:
             output_root.rmdir()
             output_root.parent.rmdir()
        except OSError:
             pass 
    except OSError as e:
        print(f"Warning: Could not cleanup temporary files: {e}")

    print(f"Final output saved to: {target_file}")
    return str(target_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove music from audio/video using Demucs.")
    parser.add_argument("input_file", help="Path to input audio or video file")
    parser.add_argument("--output", help="Path to output audio file")
    
    args = parser.parse_args()
    
    try:
        remove_music(args.input_file, args.output)
    except Exception as e:
        print(f"Error: {e}")
