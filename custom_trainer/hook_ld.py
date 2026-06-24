
import os
import sys
if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
    lib_dir = os.path.join(base, 'lib')
    if os.path.exists(lib_dir):
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = lib_dir + ':' + ld_path
