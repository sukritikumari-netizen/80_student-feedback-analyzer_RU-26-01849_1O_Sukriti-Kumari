#name = Sukriti kumari
#section = O
#ERP ID = RU-26-01849
#ps no. = 80


"""PS No. 80 - Student Feedback Analyzer."""

import argparse
import csv
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "feedback.csv"
REQUIRED_COLUMNS = ["StudentID", "Rating", "Comment"]
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "good", "has", "have", "i", "in", "is", "it", "of", "on",
    "or", "our", "the", "their", "this", "to", "too", "was", "were",
    "with", "very", "you"
}


def load_feedback(csv_path):
    """Read the feedback CSV with Pandas and verify that all required columns exist."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("CSV is missing required column(s): " + ", ".join(missing))
    return data[REQUIRED_COLUMNS].copy()


def validate_feedback(data):
    """Validate each feedback row and skip entries with missing fields or ratings outside whole numbers 1-5."""
    valid_rows, warnings = [], []
    for row_number, row in enumerate(data.to_dict("records"), start=2):
        student_id = str(row["StudentID"]).strip()
        rating_text = str(row["Rating"]).strip()
        comment = str(row["Comment"]).strip()

        if not student_id or not comment:
            warnings.append(
                f"Warning: Row {row_number} skipped - StudentID and Comment must not be empty."
            )
            continue
        if not re.fullmatch(r"[1-5]", rating_text):
            warnings.append(
                f"Warning: Row {row_number} skipped - rating must be a whole number from 1 to 5."
            )
            continue

        valid_rows.append({
            "StudentID": student_id,
            "Rating": int(rating_text),
            "Comment": comment
        })

    return pd.DataFrame(valid_rows, columns=REQUIRED_COLUMNS), warnings


def calculate_summary(data):
    """Calculate the number of valid feedback entries, average rating, and count for every rating from 5 to 1."""
    counts = {rating: int((data["Rating"] == rating).sum()) for rating in range(5, 0, -1)}
    average = float(data["Rating"].mean()) if not data.empty else 0.0
    return {"count": int(len(data)), "average": average, "rating_counts": counts}


def get_top_words(data, limit=3):
    """Count useful words in comments, remove common stop words, and return the most frequent words."""
    frequencies = Counter()
    for comment in data["Comment"].tolist():
        words = re.findall(r"[a-zA-Z]+", str(comment).lower())
        frequencies.update(
            word for word in words if len(word) > 1 and word not in STOP_WORDS
        )
    # Sort by descending frequency, then alphabetically for predictable tie handling.
    return sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:limit]


def calculate_sentiment(data):
    """Classify ratings 4-5 as positive, 3 as neutral, and 1-2 as negative; calculate percentages."""
    total = len(data)
    groups = {
        "Positive": int(data["Rating"].isin([4, 5]).sum()) if total else 0,
        "Neutral": int((data["Rating"] == 3).sum()) if total else 0,
        "Negative": int(data["Rating"].isin([1, 2]).sum()) if total else 0
    }
    return {
        name: {
            "count": count,
            "percentage": (count / total * 100) if total else 0.0
        }
        for name, count in groups.items()
    }


def display_summary(data):
    """Display valid feedback count, average, rating distribution, and the top three comment words."""
    summary = calculate_summary(data)
    print(f"Valid feedback: {summary['count']}")
    print(f"Average rating: {summary['average']:.2f}")
    print("Ratings: " + " -> ".join(
        f"{rating} -> {summary['rating_counts'][rating]}" for rating in range(5, 0, -1)
    ))
    words = get_top_words(data, 3)
    result = ", ".join(f"{word} ({count})" for word, count in words)
    print("Top 3 words: " + (result if result else "No meaningful words found"))


def display_sentiment(data):
    """Print positive, neutral, and negative percentages and display them in a Matplotlib bar chart."""
    results = calculate_sentiment(data)
    for label in ("Positive", "Neutral", "Negative"):
        item = results[label]
        print(f"{label}: {item['count']} ({item['percentage']:.2f}%)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is missing. Install it with: python3 -m pip install matplotlib")
        return

    labels = ["Positive", "Neutral", "Negative"]
    values = [results[label]["percentage"] for label in labels]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values)
    plt.title("Student Feedback Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Feedback (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{value:.2f}%", ha="center")
    plt.tight_layout()
    plt.show()


def run_analysis(csv_path, choice):
    """Load and validate the selected CSV, print row warnings, and run the chosen report."""
    raw = load_feedback(csv_path)
    clean, warnings = validate_feedback(raw)
    for warning in warnings:
        print(warning)
    if choice == "1":
        display_summary(clean)
    elif choice == "2":
        display_sentiment(clean)
    else:
        print("Invalid menu choice. Please select 1 or 2.")
        return False
    return True


def run_self_tests():
    """Run four automated assignment scenarios: normal input, one row, invalid rows, and sentiment percentages."""
    class AnalyzerTests(unittest.TestCase):
        def make_csv(self, rows):
            handle = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", newline="", encoding="utf-8", delete=False
            )
            writer = csv.writer(handle)
            writer.writerow(REQUIRED_COLUMNS)
            writer.writerows(rows)
            handle.close()
            self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
            return handle.name

        def test_case_1_normal_input(self):
            path = self.make_csv([
                ("S1", 5, "Great teaching and clear notes"),
                ("S2", 4, "Clear explanations and good examples"),
                ("S3", 2, "Too fast and notes unclear"),
                ("S4", 4, "Great notes"),
                ("S5", 3, "Good but too fast"),
                ("S6", 5, "Great feedback")
            ])
            clean, warnings = validate_feedback(load_feedback(path))
            summary = calculate_summary(clean)
            self.assertEqual(summary["count"], 6)
            self.assertAlmostEqual(summary["average"], 23 / 6, places=2)
            self.assertEqual(summary["rating_counts"], {5: 2, 4: 2, 3: 1, 2: 1, 1: 0})
            self.assertEqual(get_top_words(clean, 3), [("great", 3), ("notes", 3), ("clear", 2)])
            self.assertEqual(warnings, [])

        def test_case_2_single_row(self):
            path = self.make_csv([("S1", 5, "Great")])
            clean, warnings = validate_feedback(load_feedback(path))
            summary = calculate_summary(clean)
            self.assertEqual(summary["count"], 1)
            self.assertEqual(summary["average"], 5.0)
            self.assertEqual(summary["rating_counts"], {5: 1, 4: 0, 3: 0, 2: 0, 1: 0})
            self.assertEqual(get_top_words(clean, 3), [("great", 1)])
            self.assertEqual(warnings, [])

        def test_case_3_invalid_rows(self):
            rows = [
                ("S1", 5, "Great teaching and clear notes"),
                ("S2", 4, "Clear explanations and good examples"),
                ("S3", 2, "Too fast and notes unclear"),
                ("S4", 4, "Great notes"),
                ("S5", 3, "Good but too fast"),
                ("S6", 5, "Great feedback"),
                ("S7", 6, "Nice"),
                ("S8", "abc", "Nice")
            ]
            clean, warnings = validate_feedback(load_feedback(self.make_csv(rows)))
            self.assertEqual(len(clean), 6)
            self.assertEqual(len(warnings), 2)
            self.assertTrue(all("rating must be a whole number from 1 to 5" in w for w in warnings))
            self.assertAlmostEqual(calculate_summary(clean)["average"], 23 / 6, places=2)

        def test_case_4_sentiment(self):
            rows = [
                ("S1", 5, "Great teaching and clear notes"),
                ("S2", 4, "Clear explanations and good examples"),
                ("S3", 2, "Too fast and notes unclear"),
                ("S4", 4, "Great notes"),
                ("S5", 3, "Good but too fast"),
                ("S6", 5, "Great feedback")
            ]
            clean, _ = validate_feedback(load_feedback(self.make_csv(rows)))
            sentiment = calculate_sentiment(clean)
            self.assertAlmostEqual(sentiment["Positive"]["percentage"], 66.67, places=2)
            self.assertAlmostEqual(sentiment["Neutral"]["percentage"], 16.67, places=2)
            self.assertAlmostEqual(sentiment["Negative"]["percentage"], 16.67, places=2)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AnalyzerTests)
    )
    return result.wasSuccessful()


def main():
    """Parse optional command-line arguments, show the menu, and start the requested feedback analysis."""
    parser = argparse.ArgumentParser(description="Analyze student feedback stored in a CSV file.")
    parser.add_argument("--file", default=str(DEFAULT_CSV), help="CSV file path (default: feedback.csv)")
    parser.add_argument("--self-test", action="store_true", help="run the four built-in test cases")
    args = parser.parse_args()

    if args.self_test:
        return 0 if run_self_tests() else 1

    print("===== Student Feedback Analyzer =====")
    print("1. Summary and top 3 words")
    print("2. Sentiment")
    choice = input("Select an option (1 or 2): ").strip()
    try:
        run_analysis(args.file, choice)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
