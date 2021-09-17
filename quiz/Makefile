quiz.pdf: quiz.tex fig-E-translation-dec-5.pdf
	lualatex quiz.tex

fig-E-translation-dec-5.pdf: quiz_plots.py
	python3 quiz_plots.py

clean:
	rm -f fig-*.png
	rm -f quiz.aux
	rm -f quiz.log
	rm -f quiz.pdf

