# Build the paper PDF into paper/build/
LATEXMK ?= latexmk
MAIN    := main
OUTDIR  := build

.PHONY: all pdf watch clean view

all: pdf

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -outdir=$(OUTDIR) $(MAIN).tex

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode -outdir=$(OUTDIR) $(MAIN).tex

view: pdf
	open $(OUTDIR)/$(MAIN).pdf 2>/dev/null || xdg-open $(OUTDIR)/$(MAIN).pdf 2>/dev/null || true

clean:
	$(LATEXMK) -C -outdir=$(OUTDIR) $(MAIN).tex
	rm -rf $(OUTDIR)
