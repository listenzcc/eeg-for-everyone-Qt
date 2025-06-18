# Activate venv
./activate-python-venv.ps1

# Check the files
python "./python/find-files-with-format-check.py"

# Append the label.csv into P300-3X3 and P300-二项式 files
python "./offline-process/P300-3X3.py"
python "./offline-process/P300-binomial.py"

# Summary the latest checked results
python "./python/summary-latest-files.py"