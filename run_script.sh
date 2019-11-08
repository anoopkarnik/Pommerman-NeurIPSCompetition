#/bin/bash

echo "collecting data"
python collect_data.py

echo "running imitation model"
python imitation.py

echo "running reinforcement learning model"
python train.py

echo "starting agent"
python run.py