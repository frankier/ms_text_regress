from fractions import Fraction

import click
import numpy
import pandas
from sklearn.model_selection import train_test_split

SHORT_LETTER_SCALE = ["F", "E", "D", "C", "B", "A"]
LONG_LETTER_SCALE = [
    "F-",
    "F",
    "F+",
    "E-",
    "E",
    "E+",
    "D-",
    "D",
    "D+",
    "C-",
    "C",
    "C+",
    "B-",
    "B",
    "B+",
    "A-",
    "A",
    "A+",
]


def is_floatable(f):
    try:
        float(f)
        return True
    except ValueError:
        return False


def is_frac_str(s):
    bits = s.split("/")
    return len(bits) == 2 and is_floatable(bits[0]) and is_floatable(bits[1])


def is_barenum_str(s):
    return s.count("/") == 0 and is_floatable(s)


def is_dec_denom(s):
    bits = s.split("/")
    return len(bits) == 2 and "." in bits[1]


def drop_because(df, pred, reason):
    print(f"Dropping {pred.sum()} ({pred.mean() * 100:.2f}%) of reviews with {reason}")
    return df[~pred]


def drop_unrated(df):
    df = drop_because(df, df["review_score"].isna(), "no rating")
    df = drop_because(df, df["review_content"].isna(), "missing review")
    return df


def drop_odd_grade_types(df):
    is_any_letter = df["review_score"].isin(LONG_LETTER_SCALE)
    is_frac = df["review_score"].map(is_frac_str)
    is_barenum = df["review_score"].map(is_barenum_str)
    assert len(df[~is_frac & ~is_any_letter & ~is_barenum]) == 0
    df = drop_because(df, is_barenum, "bare number rating (i.e. no denominator)")
    is_frac_denom = df["review_score"].map(is_dec_denom)
    return drop_because(df, is_frac_denom, "fractional denominator")


def split_scores(df):
    nums = numpy.empty(len(df))
    denoms = numpy.empty(len(df))
    for idx, score in enumerate(df["review_score"]):
        if "/" in score:
            num, denom = score.split("/", 1)
            nums[idx] = float(num)
            denoms[idx] = float(denom)
        else:
            nums[idx] = nan
            denoms[idx] = nan
    df.insert(len(df.columns), "orig_num", nums)
    df.insert(len(df.columns), "orig_denom", denoms)


nan = float("nan")


def np_round(arr):
    return (arr + 0.5).astype(numpy.int32)


def put_onto_scale(df, scale):
    score_cat = pandas.Categorical(df["review_score"], categories=scale, ordered=True)
    df["label"] = score_cat.codes
    df["scale_points"] = len(scale)
    return df


def process_letter_grade_group(group_df):
    group_df["includes_zero"] = False
    group_df["multiplier"] = 1
    group_df["non_neg_error"] = False
    if group_df.iloc[0]["letter_implies_short"]:
        put_onto_scale(group_df, SHORT_LETTER_SCALE)
    else:
        put_onto_scale(group_df, LONG_LETTER_SCALE)
    return group_df


def process_includes_zero(group_df):
    multiplier = group_df.iloc[0]["multiplier"]
    includes_zero = any((label < multiplier for label in group_df["label"]))
    group_df["includes_zero"] = includes_zero
    if not includes_zero:
        group_df["label"] -= multiplier
        group_df["scale_points"] -= multiplier
    return group_df


def find_effective_nom_denom(group_df):
    if group_df.iloc[0]["is_any_letter"]:
        return process_letter_grade_group(group_df)
    else:
        group_df = common_denom_grades(group_df)
        return process_includes_zero(group_df)


def common_denom_grades(group_df):
    denoms = numpy.empty(len(group_df), dtype=numpy.int32)
    for idx, num in enumerate(group_df["orig_num"]):
        frac = Fraction.from_float(num)
        denoms[idx] = frac.limit_denominator(100).denominator
    common_denom = numpy.lcm.reduce(denoms)
    group_df["multiplier"] = common_denom
    num = common_denom * group_df["orig_num"].to_numpy()
    denom = common_denom * group_df["orig_denom"].to_numpy()
    group_df["label"] = np_round(num)
    round_denom = np_round(denom)
    group_df["non_neg_error"] = (abs(group_df["label"] - num) >= 0.05) | (
        abs(round_denom - denom) >= 0.05
    )
    group_df["scale_points"] = round_denom + 1
    return group_df


def normalize_reviews(review_df):
    print()
    # Drop unrated
    review_df = drop_unrated(review_df)

    # Strip whitespace from grades
    review_df["review_score"] = review_df["review_score"].str.replace(
        "\s+", "", regex=True
    )

    # Copy to get version to do calculations with
    working_review_df = review_df.copy()

    # Drop all rows where the review score occurs 2 or less times in the whole data set
    working_review_df = working_review_df.groupby("review_score").filter(
        lambda x: len(x) > 2
    )

    # Check/ensure that all grades are short letter, long letter, fraction or barenum
    working_review_df = drop_odd_grade_types(working_review_df)

    # Split fraction scores into numerator and denominator
    split_scores(working_review_df)

    # Divide letter scales into short and long
    # If a publisher has a mix of short and long, they're using long, otherwise short
    is_any_letter = working_review_df["review_score"].isin(LONG_LETTER_SCALE)
    is_short_letter = working_review_df["review_score"].isin(SHORT_LETTER_SCALE)
    # is_long_letter = is_any_letter & ~is_short_letter
    publisher_letter_implies_short = (
        pandas.DataFrame.from_dict(
            dict(
                publisher_name=working_review_df["publisher_name"],
                letter_implies_short=is_short_letter | ~is_any_letter,
            )
        )
        .groupby("publisher_name")
        .all()
    )
    working_review_df = working_review_df.join(
        publisher_letter_implies_short, on="publisher_name"
    )
    working_review_df["is_any_letter"] = is_any_letter

    # Now divide everything into grade types: either short letter, long letter
    # or the denominator of the fraction
    def get_grade_type(row):
        if row["is_any_letter"]:
            if row["letter_implies_short"]:
                return "short_letter"
            else:
                return "long_letter"
        else:
            return str(int(row["orig_denom"]))

    working_review_df["grade_type"] = working_review_df.apply(
        get_grade_type, axis="columns"
    )

    # Now we can filter out rare grade types
    working_review_df = working_review_df.join(
        working_review_df["grade_type"].value_counts().rename("grade_type_count"),
        on="grade_type",
    )
    working_review_df = drop_because(
        working_review_df,
        working_review_df["grade_type_count"] < 50,
        "grade type with less than 50 reviews",
    )

    # Print out some summary stats
    print("grades type counts")
    print(working_review_df["grade_type"].value_counts())
    print("unique grades", working_review_df["grade_type"].nunique())
    print("unique publishers", working_review_df["publisher_name"].nunique())
    print(
        "unique grade/publisher combinations",
        working_review_df.groupby(["grade_type", "publisher_name"]).ngroups,
    )

    # Now we can find common denominators on a (publisher, grade type) combination basis
    working_review_df = working_review_df.groupby(
        ["publisher_name", "grade_type"], group_keys=False
    ).apply(find_effective_nom_denom)
    working_review_df = drop_because(
        working_review_df, working_review_df["multiplier"] > 500, "multiplier > 500"
    )
    assert working_review_df["non_neg_error"].sum() == 0

    # More summary stats
    print("non-neg error count", working_review_df["non_neg_error"].sum())
    print("multipliers")
    print(working_review_df["multiplier"].value_counts())
    print("includes_zero")
    print(working_review_df["includes_zero"].value_counts())
    print("grade breakdown")
    print(
        working_review_df.value_counts(
            ["grade_type", "multiplier", "includes_zero", "scale_points"]
        )
    )

    # TODO: Add back in rare review_scores dropped at the beginning when they
    # are compatible with some common denominator + grade type from the same
    # publisher

    print("number of reviews left", len(working_review_df))
    print("reviews per publisher")
    print(working_review_df.value_counts(["publisher_name", "grade_type"]))

    # Delete working columns
    del working_review_df["letter_implies_short"]
    del working_review_df["is_any_letter"]
    del working_review_df["grade_type_count"]
    del working_review_df["non_neg_error"]

    return working_review_df


@click.command()
@click.argument("inf")
@click.argument("outf")
def main(inf, outf):
    normalize_reviews(pandas.read_csv(inf)).to_parquet(outf, index=False)


if __name__ == "__main__":
    main()
