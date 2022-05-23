# -*- mode: python ; coding: utf-8 -*-


block_cipher = None
site_packages = "C:\\Users\\Bogdan\\miniconda3\\envs\\setki2\\Lib\\site-packages"

a = Analysis(['SARTectron.py'],
             pathex=[
		site_packages + "\\torch\\lib",
		site_packages,
		"C:\\Users\\Bogdan\\miniconda3\\envs\\setki2\\conda-meta"
	     ],
             binaries=[],
	     datas=[
		('C:\\Users\\Bogdan\\miniconda3\\envs\\setki2\\Library\\bin\\libiomp5md.dll', '.'),
		(os.path.join(site_packages,"torch-1.6.0-py3.8.egg-info"), "torch-1.6.0-py3.8.egg-info"),
		(os.path.join(site_packages,"fvcore"), "fvcore"),
		(os.path.join(site_packages,"torch"), "torch"),
		(os.path.join(site_packages,"torchvision"), "torchvision")
	     ],
	     hiddenimports=['torch-1.6.0-py3.8.egg-info', 'fvcore', 'torch', 'torchvision'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

a.datas += Tree('D:\SARTectron\configs\Includes')

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
