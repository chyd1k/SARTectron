# -*- mode: python ; coding: utf-8 -*-


block_cipher = None
site_packages = "D:\\miniconda3\\envs\\sartectron\\Lib\\site-packages"
excludes_pkgs = []
# excludes_pkgs = ["qt5", "PyQt5", "scipy", "matplotlib", "FixTk", "tcl", "tk", "_tkinter", "tkinter", "Tkinter"]

a = Analysis(['SARTectron.py'],
             pathex=[
		site_packages + "\\torch\\lib",
		site_packages,
		"D:\\miniconda3\\envs\\sartectron\\conda-meta"
	     ],
             binaries=[],
	     datas=[
		# ('C:\\Users\\photomod\\miniconda3\\envs\\setki4\\Library\\bin\\libiomp5md.dll', '.'),
		(os.path.join(site_packages, "torch-2.12.0.dev20260222+cu128.dist-info"), "torch-2.12.0.dev20260222+cu128.dist-info"),
		(os.path.join(site_packages, "fvcore"), "fvcore"),
		(os.path.join(site_packages, "torch"), "torch"),
		(os.path.join(site_packages, "torchvision"), "torchvision"),
		(os.path.join(site_packages, "pycocotools"), "pycocotools")
	     ],
	     hiddenimports=[
		# os.path.join(site_packages, "torch-2.12.0.dev20260222+cu128.dist-info"),
		'fvcore', 'torch', 'torchvision'
             ],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=excludes_pkgs,
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

a.datas += Tree("D:\Code\SARTectron\configs\Includes")

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='SARTectron',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='SARTectron')