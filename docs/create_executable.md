# Create executable

Create an executable of dot for different OS.

## Windows

Follow these steps to generate the executable for Windows.

1. Run these commands

```
cd path/to/dot
conda activate dot
```

2. Get the path of the `site-packages` by running this command

```
python -c "import site; print(''.join(site.getsitepackages()))"
```

3. Replace `path/to/site-packages` with the path of the `site-packages` and run this command

```
pyinstaller --noconfirm --onedir --name "dot" --add-data "src/dot/fomm/config;dot/fomm/config" --add-data "src/dot/simswap/models;dot/simswap/models" --add-data "path/to/site-packages;." --add-data "configs;configs/" --add-data "data;data/" --add-data "saved_models;saved_models/" src/dot/ui/ui.py
```

The executable files can be found under the folder `dist`.

## Ubuntu

ToDo

## Mac

```
pyinstaller --noconfirm --onedir --name "dot" --add-data="src/dot/fomm/config:dot/fomm/config" --add-data="src/dot/simswap/models:dot/simswap/models" --add-data="path/to/site-packages:." --add-data="configs:configs/" --add-data="data:data/" --add-data="saved_models:saved_models/" src/dot/ui/ui.py
```

The executable files can be found under the folder `dist`.
