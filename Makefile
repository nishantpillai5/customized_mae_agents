.PHONY: clean clean_data clean_interim clean_processed lint documentation clean_documentation create_environment delete_environment requirements data test_environment sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

.ONESHELL:
SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ENV_DIR = env
PROJECT_NAME = repo
PYTHON_INTERPRETER = python

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Clean all generated data except rosbags
clean_data:
	find ./data/raw/ -type f ! \( -name "*.zip" -o -name "*.git*" \) -delete
	find ./data/interim/ -type f ! \( -name "*.zip" -o -name "*.git*" \) -delete
	find ./data/processed/ -type f ! \( -name "*.zip" -o -name "*.git*" \) -delete
	rm -rf ./data/raw/*/
	rm -rf ./data/interim/*/
	rm -rf ./data/processed/*/

## Clean all interim files
clean_interim:
	find ./data/interim/ -type f ! \( -name "*.zip" -o -name "*.git*" \) -delete
	rm -rf ./data/interim/*/

## Clean all prediction files
clean_processed:
	find ./data/processed/ -type f ! \( -name "*.zip" -o -name "*.git*" \) -delete
	rm -rf ./data/processed/*/

## Lint using isort and black (requires environment)
lint:
	isort src
	black src

## Generate documentation files (requires environment)
documentation:
	cp README.md docs/README.md
	cd docs && make html

## Clean generated documentation files
clean_documentation:
	cd docs && make clean

## Set up conda environment
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda env create --prefix ./$(ENV_DIR) --file environment.yml
	@echo ">>> New conda env created. Activate with: conda activate ./$(ENV_DIR)"
else
	@echo ">>> Conda is missing. Try using miniconda."
	# Implement venv alternative
endif

delete_environment:
	-rm -rf $(ENV_DIR)/*

## Install Python Dependencies
requirements: test_environment
	$(CONDA_ACTIVATE) ./$(ENV_DIR)
	@echo ">>> Installing pip dependencies."
	pip install -r requirements.txt
	# pip install --no-deps -r requirements_freezed.txt
	# $(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	# $(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Test python environment is setup correctly
test_environment:
	$(CONDA_ACTIVATE) ./$(ENV_DIR)
	$(PYTHON_INTERPRETER) test_environment.py

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
