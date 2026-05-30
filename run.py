"""
run.py — Cross-platform launcher (works on Windows, Mac, Linux)
Usage:  python run.py
"""
import subprocess, sys, os

os.makedirs('samples', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

print("=" * 50)
print("  QuizScanner AI")
print("=" * 50)
print("Installing dependencies...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'])

print("\nStarting server at http://localhost:5000")
print("Press Ctrl+C to stop\n")

os.chdir('src')
sys.path.insert(0, '.')
os.environ['FLASK_ENV'] = 'development'

from app import app
app.run(host='0.0.0.0', port=5000, debug=False)
