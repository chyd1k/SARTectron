import os
import sys

# Фиксим LD_LIBRARY_PATH ДО импорта scipy
if getattr(sys, 'frozen', False):
    # PyInstaller bundle
    lib_path = os.path.join(sys._MEIPASS, '.')  # binaries в корне
    os.environ['LD_LIBRARY_PATH'] = lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')

    # Добавляем в Python path
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)

print(f"BLAS LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')}")
