# Makefile for building Sphinx documentation

# Directories
SPHINX_SOURCE = source
SPHINX_BUILD = build

# Sphinx executable
SPHINX = python3 -m sphinx

# Build targets
.PHONY: html clean

html:
	@( \
		echo "Changing to $(SPHINX_SOURCE) directory"; \
		cd $(SPHINX_SOURCE); \
		echo "Building HTML documentation"; \
		SPHINX_QUICK_BUILD=$(QUICKBUILD) $(SPHINX) -T -b html -d ../$(SPHINX_BUILD)/doctrees -D language=en . ../$(SPHINX_BUILD)/html; \
	)

clean:
	@rm -rf $(SPHINX_BUILD)/*
	@rm -rf $(SPHINX_SOURCE)/auto_examples
	@rm -rf $(SPHINX_SOURCE)/auto_api
	@rm -rf $(SPHINX_SOURCE)/gen_modules
	@rm -rf $(SPHINX_SOURCE)/sg_execution_times.rst
