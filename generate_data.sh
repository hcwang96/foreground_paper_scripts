#!/bin/bash

echo "Starting first case"
python first_case.py 1e-5
python first_case.py 1e-4
python first_case.py 1e-3

echo "Starting second case"
python second_case.py 1e-5
python second_case.py 1e-4
python second_case.py 1e-3

echo "Starting third case"
python third_case.py 1e-5
python third_case.py 1e-4
python third_case.py 1e-3