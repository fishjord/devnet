writeup.pdf : writeup.ps
	ps2pdf writeup.ps writeup.pdf

writeup.ps : writeup.dvi
	dvips -o writeup.ps -G0 -tletter writeup.dvi

writeup.dvi : writeup.tex
	latex writeup.tex