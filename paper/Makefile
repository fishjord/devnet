ijcnn_submitted.pdf : ijcnn_submitted.ps
	ps2pdf ijcnn_submitted.ps ijcnn_submitted.pdf

ijcnn_submitted.ps : ijcnn_submitted.dvi
	dvips -o ijcnn_submitted.ps -G0 -tletter ijcnn_submitted.dvi

ijcnn_submitted.dvi : ijcnn_submitted.tex
	latex ijcnn_submitted.tex