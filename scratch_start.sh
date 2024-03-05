#!/bin/bash

gnome-terminal -- bash -c "echo '***Start scratch_server***'; cd ~/akari_scratch; source venv/bin/activate; python3 -m akari_scratch_server.cli; bash";

echo '***Start scratch-gui***';
cd ~/akari_scratch/scratch-gui;
yarn start;
