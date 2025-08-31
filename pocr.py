#!/usr/bin/env python3
from paddleocr import PaddleOCR
import fitz
import cv2
import numpy as np
import tqdm
import sys
import argparse
import pathlib

def rotate_point(point, angle, center):
    """Rotate a point around center by angle degrees."""
    import math
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x, y = point
    cx, cy = center
    x -= cx
    y -= cy
    new_x = x * cos_a - y * sin_a + cx
    new_y = x * sin_a + y * cos_a + cy
    return (new_x, new_y)



# remove rotate_point(...) entirely

def process_pdf(input_pdf_path, output_pdf_path):
    pdf_doc = fitz.open(input_pdf_path)
    img = fitz.open()
    pure = fitz.open() if args.pure else None

    ocr = PaddleOCR(use_textline_orientation=True, lang=args.lang)

    for page_number in tqdm.tqdm(range(pdf_doc.page_count)):
        page = pdf_doc.load_page(page_number)

        # strip existing text from the source page before extracting images
        page.add_redact_annot(page.rect)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        images = page.get_images(full=True)
        if not images:
            # optional: preserve pagination by copying original page
            # newp = img.new_page(width=page.mediabox.width, height=page.mediabox.height)
            # newp.show_pdf_page(newp.rect, pdf_doc, page_number)
            continue

        xref_0 = images[0][0]
        bin_img = pdf_doc.extract_image(xref_0)
        cim = cv2.imdecode(np.frombuffer(bin_img["image"], np.uint8), cv2.IMREAD_COLOR)

        if args.cv:
            cv2.imshow(sys.argv[0], cim); cv2.waitKey(1)

        # target pages sized by the UNROTATED mediabox
        w, h = page.mediabox.width, page.mediabox.height
        img_page = img.new_page(width=w, height=h)
        if args.pure:
            pure_page = pure.new_page(width=cim.shape[1], height=cim.shape[0])  # pixel space

        # insert the scan as a full-page image (no rotation, rect is unrotated)
        img_page.insert_image(img_page.rect, stream=bin_img["image"])

        # if the source image had /Decode, consider copying it over instead of deleting
        if "Decode" in pdf_doc.xref_get_keys(xref_0):
            t, v = pdf_doc.xref_get_key(xref_0, "Decode")
            if t and v:
                img.xref_set_key(img_page.get_images()[0][0], "Decode", v)
        else:
            # remove explicit Decode if any was attached by insertion
            img.xref_set_key(img_page.get_images()[0][0], "Decode", "null")

        # OCR
        pred = ocr.predict(cim)
        res = pred[0] if isinstance(pred, list) and pred else pred
        if not res or not res.get('rec_texts'):
            continue

        # scale pixel -> page (mediabox) coords
        def to_page_xy(pt):
            return (
                float(pt[0]) / cim.shape[1] * w,
                float(pt[1]) / cim.shape[0] * h,
            )

        for word, bbox, conf in zip(res['rec_texts'], res['rec_polys'], res['rec_scores']):
            if conf < 0.9:
                continue

            # rectangle in UNROTATED page space
            x0, y0 = to_page_xy(bbox[0])
            x1, y1 = to_page_xy(bbox[2])
            R = fitz.Rect(x0, y0, x1, y1)

            fn = "helv" if word.isascii() else "china-s"
            fs = R.width / fitz.get_text_length(word, fn, 1)

            # IMPORTANT: derotate insertion point, do NOT pass rotate=page.rotation
            p = fitz.Point(R.x0, R.y1) * page.derotation_matrix
            img_page.insert_text(p, word, fontname=fn, fontsize=fs, render_mode=3)

            if args.pure:
                # place in pixel space directly
                px0 = min(p[0] for p in bbox)
                py0 = min(p[1] for p in bbox)
                px1 = max(p[0] for p in bbox)
                py1 = max(p[1] for p in bbox)
                Rp = fitz.Rect(px0, py0, px1, py1)
                fs_p = Rp.width / fitz.get_text_length(word, fn, 1)
                pure_page.insert_text((Rp.x0, Rp.y1), word, fontname=fn, fontsize=fs_p, render_mode=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program that adds hidden(but copiable) text layer to image pdf.",
        epilog="Copyright (C) 2024 Cao Yang. This is free software; distributed under GPLv3. There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.",
    )
    parser.add_argument("input_file", help="Input PDF file")
    parser.add_argument("output_file", help="Output PDF file")
    parser.add_argument("-p", "--pure", action="store_true")
    parser.add_argument("-c", "--cv", action="store_true")
    parser.add_argument("-l", "--lang", default="en")

    args = parser.parse_args()
    if args.cv:
        cv2.namedWindow(sys.argv[0], cv2.WINDOW_NORMAL)
    process_pdf(args.input_file, args.output_file)
