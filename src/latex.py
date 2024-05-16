# Utility routines

import numpy as np
import pandas as pd


def write_table(
    df,
    categories,
    columns,
    ranges,
    range_column,
    title="",
    caption="",
    label="",
    df2=None,
    columns2=None,
    ranges2=None,
):
    num_columns = len(columns)
    format_str = " c c "
    sub_columns = 2
    column_labels = " & $\\mu$ & $\\sigma$"
    category_columns = len(categories)

    column_format = "\\begin{tabular}{@{} l "
    title_str1 = "  "
    title_str2 = categories[0]["name"]
    for i in range(1, category_columns):
        column_format += " l "
        title_str1 += " & "
        title_str2 += f" & {categories[i]['name']}"

    for i in range(num_columns):
        column_format += format_str
        title_str1 += f" & \\multicolumn{{{sub_columns}}}{{c}}{{ {columns[i]} }}"
        title_str2 += column_labels

    if df2 is not None:
        num_columns2 = len(columns2)
        for i in range(num_columns2):
            column_format += format_str
            title_str1 += f" & \\multicolumn{{{sub_columns}}}{{c}}{{ {columns2[i]} }}"
            title_str2 += column_labels

    cmidrule = " \\cmidrule{2-" + str(2 + num_columns * sub_columns + 1) + "}"

    title_str1 += "\\\\"
    title_str2 += "\\\\"
    column_format += " @{}}"

    print("%%%%%")
    print(f"% Table Data ({title})")
    print("%")
    print("\\begin{table*}")
    print(f"\\caption{{ {caption} }}")
    print(f"\\label{{ {label} }}")
    print("\\begin{center}")
    print(column_format)
    print("\\toprule")

    print(title_str1)
    print(title_str2)
    print("\\midrule")

    for cat in categories[0]["labels"]:

        if categories[0]["proper_name"] is not None:
            cat_label = categories[0]["proper_name"][cat]
        else:
            cat_label = str(cat)

        for index, sub_cat in enumerate(categories[1]["labels"]):

            if categories[1]["proper_name"] is not None:
                sub_cat_label = categories[1]["proper_name"][sub_cat]
            else:
                sub_cat_label = str(sub_cat)

            for sub_sub_cat in categories[2]["labels"]:

                if categories[2]["proper_name"] is not None:
                    sub_sub_cat_label = categories[2]["proper_name"][sub_sub_cat]
                else:
                    sub_sub_cat_label = str(sub_sub_cat)

                if cat == "all":
                    df_slice = df[
                        (df[categories[1]["column"]] == sub_cat) & (df[categories[2]["column"]] == sub_sub_cat)
                    ]
                else:
                    df_slice = df[
                        (df[categories[0]["column"]] == cat)
                        & (df[categories[1]["column"]] == sub_cat)
                        & (df[categories[2]["column"]] == sub_sub_cat)
                    ]

                if df2 is not None:
                    if cat == "all":
                        df_slice2 = df2[
                            (df2[categories[1]["column"]] == sub_cat) & (df2[categories[2]["column"]] == sub_sub_cat)
                        ]
                    else:
                        df_slice2 = df2[
                            (df2[categories[0]["column"]] == cat)
                            & (df2[categories[1]["column"]] == sub_cat)
                            & (df2[categories[2]["column"]] == sub_sub_cat)
                        ]
                else:
                    df_slice2 = None

                s = f"{cat_label} & {sub_cat_label} & {sub_sub_cat_label}"

                for col, ran in zip(columns, ranges):
                    if ran == "all":
                        s += " & " + f"{(df_slice[col].mean()):5.1f} & {(df_slice[col].std()):5.2f}"
                    elif ran == "min":
                        s += " & " + f"{(df_slice[col].max()):5.1f} & {(df_slice[col].min()):5.2f}"

                if df2 is not None:
                    for col, ran in zip(columns2, ranges2):
                        if ran == "all":
                            s += " & " + f"{(df_slice2[col].mean()):5.1f} & {(df_slice2[col].std()):5.2f}"
                        elif ran == "min":
                            s += " & " + f"{(df_slice2[col].max()):5.1f} & {(df_slice2[col].min()):5.2f}"

                s += "\\\\"
                print(s)

                cat_label = ""
                sub_cat_label = ""

            if index < len(categories[1]["labels"]) - 1:
                print(cmidrule)

        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table*}")
    print("%")
    print("%%%%%")
