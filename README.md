PS No. 80 - Student Feedback Analyzer
=====================================

OVERVIEW
This Python 3 console program reads feedback.csv, validates each rating,
calculates summary statistics, finds the top three meaningful comment words,
and displays sentiment percentages with a Matplotlib bar chart.

FILES
one.py             Main Python program
feedback.csv       Six-row sample input
requirements.txt   Required packages
README.txt         Instructions

INSTALL
python3 -m pip install -r requirements.txt

RUN
python3 one.py
Choose 1 for Summary and top 3 words, or 2 for Sentiment.

Use a different CSV:
python3 one.py --file path/to/feedback.csv

TEST THE FOUR ASSIGNMENT CASES
python3 one.py --self-test

CSV FORMAT
StudentID,Rating,Comment
S1,5,Great teaching and clear notes

Ratings must be whole numbers from 1 to 5. Rows with missing required
values or invalid ratings are skipped with a warning.

SENTIMENT RULES
Positive = ratings 4 and 5
Neutral = rating 3
Negative = ratings 1 and 2

WORD FREQUENCY
Words are converted to lowercase; common stop words such as "the", "and",
"to", "too", and "but" are excluded. Frequency ties are sorted alphabetically.
