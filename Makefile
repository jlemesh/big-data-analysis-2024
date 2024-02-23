task_optional_0223:
	xelatex -output-directory=build task_optional_0223.tex
	biber build/task_optional_0223
	xelatex -output-directory=build task_optional_0223.tex
	open build/task_optional_0223.pdf || xdg-open build/task_optional_0223.pdf

ubuntu:
	echo "Diegiamas LaTeX (PdfLaTeX, XeTeX ir kt.)"
	sudo apt-get install texlive-full
	echo "Diegiama literatūros sąrašo tvarkyklė Biber skirta BibLaTeX paketui"
	sudo apt-get install biber
	echo "Diegiami OpenType šriftai"
	sudo apt-get install fonts-texgyre
	echo "Diegiamas Palemonas šriftas į sistemą"
	sudo cp -r Palemonas-2.1 /usr/share/fonts/truetype/

clean:
	git clean -dfx
