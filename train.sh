#!/bin/sh

set -x

python ppnet_main.py --orth-coef=0 --name "orth_coef_0"
python ppnet_main.py --sep-coef=0 --name "sep_coef_0"
python ppnet_main.py --name "full"
