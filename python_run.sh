#!/bin/bash

echo "Enter name of .py file to run: "
read PYTHONFILE

echo "Version:"
python --version
echo "Running file ${PYTHONFILE}..."
echo
time python3 ${PYTHONFILE}
echo
echo "${PYTHONFILE} was run successfully."
