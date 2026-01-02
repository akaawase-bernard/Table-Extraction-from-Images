


"""
Table Extraction from Images (No Grid Lines)
Author: Bernard

Geometry-based OCR table extraction with semantic cleanup.
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt


def _is_number_like(s):
    return bool(re.search(r"\d", s))


def _is_unit_like(s):
    return bool(re.search(r"[()\-/¹²³]", s))


def extract_table_from_image(
    image_path,
    *,
    row_tol=12,
    col_tol=25,
    show_figure=True,
    csv_path=None,
    figure_path=None,   # <<< ADD THIS
):
    # -------------------------------------------------
    # 1. Load image
    # -------------------------------------------------
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------
    # 2. OCR
    # -------------------------------------------------
    ocr = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DICT,
        config="--psm 6"
    )

    words = []
    for i in range(len(ocr["text"])):
        txt = ocr["text"][i].strip()
        if txt:
            words.append({
                "text": txt,
                "x": ocr["left"][i],
                "y": ocr["top"][i],
                "w": ocr["width"][i],
                "h": ocr["height"][i],
                "cx": ocr["left"][i] + ocr["width"][i] / 2,
                "cy": ocr["top"][i] + ocr["height"][i] / 2,
            })

    # -------------------------------------------------
    # 3. Group into rows
    # -------------------------------------------------
    words = sorted(words, key=lambda d: d["cy"])
    rows = []

    for w in words:
        for row in rows:
            if abs(w["cy"] - row[0]["cy"]) < row_tol:
                row.append(w)
                break
        else:
            rows.append([w])

    for row in rows:
        row.sort(key=lambda d: d["cx"])

    # -------------------------------------------------
    # 4. Infer columns
    # -------------------------------------------------
    x_centers = sorted([w["cx"] for w in words])
    col_centers = []
    for x in x_centers:
        if not col_centers or abs(x - col_centers[-1]) > col_tol:
            col_centers.append(x)

    # -------------------------------------------------
    # 5. Raw table
    # -------------------------------------------------
    raw = []
    for row in rows:
        cells = [""] * len(col_centers)
        for w in row:
            j = np.argmin([abs(w["cx"] - c) for c in col_centers])
            cells[j] += (" " if cells[j] else "") + w["text"]
        raw.append(cells)

    df = pd.DataFrame(raw)

    # -------------------------------------------------
    # 6. REMOVE FAKE COLUMNS
    # -------------------------------------------------
    keep = []
    for c in df.columns:
        col = df[c].iloc[1:].astype(str).str.strip()
        non_empty = col[col != ""]

        if len(non_empty) == 0:
            continue

        if non_empty.apply(_is_number_like).any():
            keep.append(c)
            continue

        if non_empty.nunique() > 1:
            keep.append(c)
            continue

    df = df[keep].reset_index(drop=True)

    # -------------------------------------------------
    # 7. MERGE SPLIT TEXT COLUMNS
    # -------------------------------------------------
    merged = []
    skip = False

    for i in range(df.shape[1]):
        if skip:
            skip = False
            continue

        col = df.iloc[:, i].astype(str)

        if i < df.shape[1] - 1:
            next_col = df.iloc[:, i + 1].astype(str)

            if (
                col.iloc[1:].str.isalpha().any()
                and next_col.iloc[1:].str.isalpha().any()
            ):
                merged.append(col.str.strip() + " " + next_col.str.strip())
                skip = True
                continue

        merged.append(col)

    df = pd.concat(merged, axis=1)
    df.columns = range(df.shape[1])

    # -------------------------------------------------
    # 8. Save CSV
    # -------------------------------------------------
    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved CSV → {csv_path}")

    # -------------------------------------------------
    # 9. Visualization (vertical) + SAVE FIGURE
    # -------------------------------------------------
    if show_figure or figure_path:
        overlay = img.copy()
        for w in words:
            cv2.rectangle(
                overlay,
                (int(w["x"]), int(w["y"])),
                (int(w["x"] + w["w"]), int(w["y"] + w["h"])),
                (0, 255, 0),
                1,
            )

        for x in col_centers:
            cv2.line(
                overlay,
                (int(x), 0),
                (int(x), overlay.shape[0]),
                (255, 0, 0),
                2,
            )

        fig, ax = plt.subplots(2, 1, figsize=(10, 6))

        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Input Image", fontsize=30)
        #ax[0].axis("off")

        ax[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax[1].set_title("Detected Text & Columns", fontsize=30)
        ax[1].axis("off")

        plt.tight_layout()

        if figure_path:
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            plt.savefig(figure_path, dpi=300, bbox_inches="tight")
            print(f"[INFO] Saved figure:  {figure_path}")

        if show_figure:
            plt.show()
        else:
            plt.close(fig)

    return df


# ------------------ TEST ------------------
if __name__ == "__main__":
    df = extract_table_from_image(
        "data/table.png",
        csv_path="output/table_clean.csv",
        figure_path="output/table_debug.png",
        show_figure=True,
    )

    print("\nThe TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


# """
# Table Extraction from Images (No Grid Lines)
# Author: Bernard

# Geometry-based OCR table extraction with semantic cleanup.
# """

# import os
# import re
# import cv2
# import numpy as np
# import pandas as pd
# import pytesseract
# import matplotlib.pyplot as plt


# def _is_number_like(s):
#     return bool(re.search(r"\d", s))


# def _is_unit_like(s):
#     return bool(re.search(r"[()\-/¹²³]", s))


# def extract_table_from_image(
#     image_path,
#     *,
#     row_tol=12,
#     col_tol=25,
#     show_figure=True,
#     csv_path=None,
# ):
#     # -------------------------------------------------
#     # 1. Load image
#     # -------------------------------------------------
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Could not load image: {image_path}")

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # -------------------------------------------------
#     # 2. OCR
#     # -------------------------------------------------
#     ocr = pytesseract.image_to_data(
#         gray,
#         output_type=pytesseract.Output.DICT,
#         config="--psm 6"
#     )

#     words = []
#     for i in range(len(ocr["text"])):
#         txt = ocr["text"][i].strip()
#         if txt:
#             words.append({
#                 "text": txt,
#                 "x": ocr["left"][i],
#                 "y": ocr["top"][i],
#                 "w": ocr["width"][i],
#                 "h": ocr["height"][i],
#                 "cx": ocr["left"][i] + ocr["width"][i] / 2,
#                 "cy": ocr["top"][i] + ocr["height"][i] / 2,
#             })

#     # -------------------------------------------------
#     # 3. Group into rows
#     # -------------------------------------------------
#     words = sorted(words, key=lambda d: d["cy"])
#     rows = []

#     for w in words:
#         for row in rows:
#             if abs(w["cy"] - row[0]["cy"]) < row_tol:
#                 row.append(w)
#                 break
#         else:
#             rows.append([w])

#     for row in rows:
#         row.sort(key=lambda d: d["cx"])

#     # -------------------------------------------------
#     # 4. Infer columns
#     # -------------------------------------------------
#     x_centers = sorted([w["cx"] for w in words])
#     col_centers = []
#     for x in x_centers:
#         if not col_centers or abs(x - col_centers[-1]) > col_tol:
#             col_centers.append(x)

#     # -------------------------------------------------
#     # 5. Raw table
#     # -------------------------------------------------
#     raw = []
#     for row in rows:
#         cells = [""] * len(col_centers)
#         for w in row:
#             j = np.argmin([abs(w["cx"] - c) for c in col_centers])
#             cells[j] += (" " if cells[j] else "") + w["text"]
#         raw.append(cells)

#     df = pd.DataFrame(raw)

#     # -------------------------------------------------
#     # 6. REMOVE FAKE COLUMNS
#     # -------------------------------------------------
#     keep = []
#     for c in df.columns:
#         col = df[c].iloc[1:].astype(str).str.strip()
#         non_empty = col[col != ""]

#         if len(non_empty) == 0:
#             continue

#         if non_empty.apply(_is_number_like).any():
#             keep.append(c)
#             continue

#         if non_empty.nunique() > 1:
#             keep.append(c)
#             continue

#     df = df[keep].reset_index(drop=True)

#     # -------------------------------------------------
#     # 7. MERGE SPLIT TEXT COLUMNS
#     # -------------------------------------------------
#     merged = []
#     skip = False

#     for i in range(df.shape[1]):
#         if skip:
#             skip = False
#             continue

#         col = df.iloc[:, i].astype(str)

#         if i < df.shape[1] - 1:
#             next_col = df.iloc[:, i + 1].astype(str)

#             if (
#                 col.iloc[1:].str.isalpha().any()
#                 and next_col.iloc[1:].str.isalpha().any()
#             ):
#                 merged.append(col.str.strip() + " " + next_col.str.strip())
#                 skip = True
#                 continue

#         merged.append(col)

#     df = pd.concat(merged, axis=1)
#     df.columns = range(df.shape[1])

#     # -------------------------------------------------
#     # 8. Save CSV
#     # -------------------------------------------------
#     if csv_path:
#         os.makedirs(os.path.dirname(csv_path), exist_ok=True)
#         df.to_csv(csv_path, index=False)
#         print(f"[INFO] Saved CSV → {csv_path}")

#     # -------------------------------------------------
#     # 9. Visualization (vertical)
#     # -------------------------------------------------
#     if show_figure:
#         overlay = img.copy()
#         for w in words:
#             cv2.rectangle(
#                 overlay,
#                 (int(w["x"]), int(w["y"])),
#                 (int(w["x"] + w["w"]), int(w["y"] + w["h"])),
#                 (0, 255, 0),
#                 1,
#             )

#         for x in col_centers:
#             cv2.line(overlay, (int(x), 0), (int(x), overlay.shape[0]), (255, 0, 0), 2)

#         fig, ax = plt.subplots(2, 1, figsize=(10, 6))
#         ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         ax[0].set_title("Original Image", fontsize = 30)
#         #ax[0].axis("off")

#         ax[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#         ax[1].set_title("Detected Text & Columns", fontsize = 30)
#         ax[1].axis("off")
#         plt.tight_layout()
#         plt.show()

#     return df


# # ------------------ TEST ------------------
# if __name__ == "__main__":
#     df = extract_table_from_image(
#         "data/table.png",
#         csv_path="output/table_clean.csv",
#         show_figure=True,
#     )

#     print("\nCLEANED TABLE")
#     print("=" * 80)
#     print(df.to_string(index=False))
#     print("=" * 80)






