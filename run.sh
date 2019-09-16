#!/bin/bash

PYTHONFILE=intro.py

echo "Version:"
python --version
echo "Running file ${PYTHONFILE}..."
echo
time python3 ${PYTHONFILE}
echo
echo "${PYTHONFILE} was run successfully."
