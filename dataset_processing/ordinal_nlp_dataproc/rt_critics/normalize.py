from fractions import Fraction

import click
import numpy
import pandas

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

# These reviews have odd grades which throw off the normalization resulting in a
# large number of scale points. We just remove them manually.
# rotten_tomatoes_link, publisher_name, critic_name, review_score
BLACKLIST = [
    (
        "m/the_humbling",
        "Spirituality & Practice",
        "Frederic and Mary Ann Brussat",
        "2.7/5",
    ),
    ("m/the_number_23", "ComingSoon.net", "Edward Douglas", "2.3/10"),  # Clearly a joke
    ("m/the_dark_knight_rises", "3AW", "Jim Schembri", "4.4/5"),
    (
        "m/sherlock_holmes_2009",
        "Northwest Herald (Crystal Lake, IL)",
        "Jeffrey Westhoff",
        "2.4/4",
    ),
    (
        "m/chloe_in_the_afternoon",
        "Combustible Celluloid",
        "Jeffrey M. Anderson",
        "3.4/4",
    ),
    (
        "m/heaven_is_for_real",
        "Commercial Appeal (Memphis, TN)",
        "John Beifuss",
        "2.4/4",
    ),
    ("m/undefeated_2012", "MovieFreak.com", "Sara Michelle Fetters", "3.4/4"),
    ("m/nebraska", "Herald Sun (Australia)", "Leigh Paatsch", "4.4/5"),
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


def is_gt1_frac(s):
    bits = s.split("/")
    try:
        n = float(bits[0])
        d = float(bits[1])
    except ValueError:
        return False
    return n > d


def drop_because(df, pred, reason):
    print(f"Dropping {pred.sum()} ({pred.mean() * 100:.2f}%) of reviews with {reason}")
    return df[~pred]


def is_click_here(s):
    return s.str.match(
        r"(click (to |for |here)|full review (in|at) |\(no quote available.\)|see website |podcast review|to read review).*",
        case=False,
    )


def is_grade(s):
    norm_s = s.strip().upper()
    return norm_s in LONG_LETTER_SCALE or is_frac_str(norm_s) or is_floatable(norm_s)


def drop_unrated(df):
    df = drop_because(df, df["review_score"].isna(), "no rating")
    df = drop_because(df, df["review_content"].isna(), "missing review")
    df = drop_because(df, is_click_here(df["review_content"]), "review is 'click here'")
    df = drop_because(df, df["review_content"].map(is_grade), "review is grade")
    df = drop_because(
        df, df["review_content"].str.len() == 1, "single character review"
    )

    return df


def drop_odd_grade_types(df):
    is_any_letter = df["review_score"].isin(LONG_LETTER_SCALE)
    is_frac = df["review_score"].map(is_frac_str)
    is_barenum = df["review_score"].map(is_barenum_str)
    assert len(df[~is_frac & ~is_any_letter & ~is_barenum]) == 0
    df = drop_because(df, is_barenum, "bare number rating (i.e. no denominator)")
    is_frac_denom = df["review_score"].map(is_dec_denom)
    df = drop_because(df, is_frac_denom, "fractional denominator")
    is_gt1 = df["review_score"].map(is_gt1_frac)
    return drop_because(df, is_gt1, "numerator > denominator")


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


def normalize_review_text(text):
    import re

    text = text.replace(r"\'", "'")
    return re.sub(
        r"%u([a-fA-F0-9]{4}|[a-fA-F0-9]{2})", lambda m: chr(int(m.group(1), 16)), text
    )


def is_non_english(s):
    from ftlangdetect import detect

    return len(s) >= 40 and detect(s)["lang"] != "en"


def normalize_reviews(review_df):
    print()
    # Drop unrated
    review_df = drop_unrated(review_df)

    # Drop complete duplicates
    review_df.drop_duplicates(
        [
            "critic_name",
            "publisher_name",
            "review_content",
            "review_score",
            "rotten_tomatoes_link",
        ],
        inplace=True,
    )

    blacklist_bool = numpy.zeros(review_df.shape[0], dtype=bool)
    for rotten_tomatoes_link, publisher_name, critic_name, review_score in BLACKLIST:
        matches = (
            (review_df["rotten_tomatoes_link"] == rotten_tomatoes_link)
            & (review_df["publisher_name"] == publisher_name)
            & (review_df["critic_name"] == critic_name)
            & (review_df["review_score"] == review_score)
        )
        if not any(matches):
            print(
                f"Blacklist entry not found: {rotten_tomatoes_link}, {publisher_name}, {critic_name}, {review_score}"
            )
        blacklist_bool |= matches
    review_df = drop_because(review_df, blacklist_bool, "blacklisted items")

    # Normalize review text
    review_df["review_content"] = review_df["review_content"].map(normalize_review_text)

    # Filter out non-English
    review_df = drop_because(
        review_df, review_df["review_content"].map(is_non_english), "non-English review"
    )

    # Strip whitespace from grades
    review_df["review_score"] = review_df["review_score"].str.replace(
        r"\s+", "", regex=True
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
    working_review_df = drop_because(
        working_review_df, working_review_df["scale_points"] > 255, "scale_points > 255"
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
