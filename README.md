# CSE_240D-DCQ

## Warning
**This code is based on an obsolete project called [*distiller*](https://github.com/IntelLabs/distiller)**, we managed to modified the source code to let it run on latest version of python. 

## Requirement 
- Python 3.10.12
- virtualenv

## Instruction
1. run `virtualenv env` to create a python virtual environment
2. run `source env/bin/activate` to activate python virtual environment
3. run `pip install -r requirements.txt` to install all dependencies
4. follows the commands in `cmd` in run Floating point training, quantized 32/1 bit training, quantized 1 bit training with DCQ, and fine tuning.

## Source Code
- Training script in `./` directory
- models in `./src/models`
- quantization in `./src/distiller/quantization`

## Issues
It is currently known that non-deterministic  results may occur when running the same command. If the results produced are completely different from expectations, it is recommended to execute the command multiple times.

## Contact
- Contact jid001@ucsd.edu if you have any problem.

