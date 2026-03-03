import sys
import os
import subprocess

# ── Environment Auto-Detection ───────────────────────────────────────────────
# If we have an 'env' directory, use its python to avoid ModuleNotFoundErrors
def _check_venv():
    root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(root, "env", "Scripts", "python.exe")
    if os.path.exists(venv_python) and sys.executable.lower() != venv_python.lower():
        print(f"[*] Re-launching using virtual environment: {venv_python}")
        # Pass all arguments to the new process
        result = subprocess.run([venv_python] + sys.argv)
        sys.exit(result.returncode)

if __name__ == "__main__":
    _check_venv()
    
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from app.main import main
    main()
