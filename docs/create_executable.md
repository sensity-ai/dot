# Create executable

Create an executable of dot for different OS.

## Windows

Run these commands

```
cd path/to/dot
conda activate dot
pyinstaller --noconfirm --onedir --name "dot" --add-data "src/dot/fomm/config;dot/fomm/config" --add-data "src/dot/simswap/models;dot/simswap/models" --add-data "path/to/Anaconda3/envs/dot/Lib/site-packages;." --add-data "configs;configs/" --add-data "data;data/" --add-data "saved_models;saved_models/" src/dot/ui/ui.py
```

The executable files can be found under the folder `dist`.

## Ubuntu

ToDo

## Mac

ToDo
