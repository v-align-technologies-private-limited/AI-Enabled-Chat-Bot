#!/bin/bash
#This script activates the conda environment and runs the Python script

#Navigate to the project directory
cd /home/ubuntu/AI-Enabled-Chat-Bot || exit

#Activate the conda environment
source /home/ubuntu/anaconda3/bin/activate ai_enabled_chat

#Run the Python script
python app.py
