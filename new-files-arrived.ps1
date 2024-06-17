# Activate venv
./activate-python-venv.ps1

# Check the files
python "./python/find-files-with-format-check.py"

# Summary the latest checked results
python "./python/summary-latest-files.py"