#!/bin/bash

printf "\033[0;32m Launching flake8 \033[0m\n"
flake8 \
--max-line-length 250 \
--ignore ANN101,W503,E722 vilu/