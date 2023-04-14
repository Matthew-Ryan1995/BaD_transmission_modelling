.ONESHELL:

SHELL=/bin/zsh
# Update table and python code based on parameters in parameter_ranges.json
P=python3
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

SOURCE:=data/parameter_ranges.json

FILES=code/01_sir_with_masks.py code/write_params_to_latex.py

all: prog

source: $(SOURCE)

prog: source
	$(CONDA_ACTIVATE) my-env; $(P) $(FILES)
	
