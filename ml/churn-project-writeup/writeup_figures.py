"""The figures quoted in README.md, with the file in ml-end-to-end each one comes from."""

FIGURES: dict[str, tuple[str, str]] = {
    "7,043 customers": ("customers in the dataset", "docs/data_card.md"),
    "5,634 training rows": ("training split", "reports/final_test.md"),
    "1,409 test rows": ("test split", "reports/final_test.md"),
    "0.674": ("logistic cross-validation PR-AUC", "reports/final_test.md"),
    "0.665": ("logistic test PR-AUC", "reports/final_test.md"),
    "0.662": ("forest test PR-AUC", "reports/final_test.md"),
    "0.663": ("LightGBM test PR-AUC", "reports/final_test.md"),
    "275 of the 374": ("churners caught at cutoff 0.33", "reports/final_test.md"),
    "493 flagged": ("flagged at cutoff 0.33", "reports/final_test.md"),
}
