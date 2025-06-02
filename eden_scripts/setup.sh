#!/bin/bash

# setup.sh
# This script sets up a Python virtual environment and installs dependencies


VENV_NAME="venv"
REQUIREMENTS_FILE="requirements.txt"
PYTHON_SCRIPT="download_datasets.py"

echo "Starting setup..."

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: ${VENV_NAME}..."
    python3 -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment. Do you have python3 installed?"
        exit 1
    fi
else
    echo "Virtual environment '${VENV_NAME}' already exists. Skipping creation."
fi

echo "Activating virtual environment: ${VENV_NAME}..."
source "${VENV_NAME}/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

echo "Installing dependencies from ${REQUIREMENTS_FILE}..."
pip install -r "$REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. Please check your '${REQUIREMENTS_FILE}'."
    deactivate
    exit 1
fi
echo "Dependencies installed successfully."

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: '${PYTHON_SCRIPT}' not found in the current directory."
    echo "Please ensure '${PYTHON_SCRIPT}' is present."
    deactivate
    exit 1
fi

echo "Process completed."