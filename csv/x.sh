#!/bin/bash

for file in monks.*; do
    if [ -f ${file} ]; then
        mv ${file} ${file}.csv
    fi
done
