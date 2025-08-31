#!/usr/bin/env python3
from paddleocr import PaddleOCR
import fitz
import cv2
import numpy as np
import tqdm
import sys
import logging
import argparse
import pathlib

logging.disable(logging.DEBUG)


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


def process_pdf(input_pdf_path, output_pdf_path):
    pdf_doc = fitz.open(input_pdf_path)
    if args.pure:
        pure = fitz.open()
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang=args.lang,
    )
    for page_number in tqdm.tqdm(
        range(
            pdf_doc.page_count
            # 1
            # 10
        )
    ):
        page = pdf_doc.load_page(page_number)
        page.add_redact_annot(page.rect)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        image_list = page.get_images()
        if not image_list:
            continue
        xref_0 = image_list[0][0]
        bin_img = pdf_doc.extract_image(xref_0)
        cim = cv2.imdecode(np.frombuffer(bin_img["image"], np.uint8), cv2.IMREAD_COLOR)
        # Do not rotate cim, as the page image is not rotated, and text is inserted over it
        if args.cv:
            cv2.imshow(sys.argv[0], cim)
            cv2.waitKey(1)
        if args.no_ocr:
            continue
        if args.pure:
            new_page = pure.new_page(width=cim.shape[1], height=cim.shape[0])
        text = ocr.predict(cim)
        if isinstance(text, list) and text:
            result = text[0]
        else:
            result = text
        if not result['rec_texts']:
            continue
        for i in range(len(result['rec_texts'])):
            word = result['rec_texts'][i]
            bbox = result['rec_polys'][i]
            conf = result['rec_scores'][i]
            if conf < 0.9:
                continue

            # Rotate bbox back to unrotated coordinates if page is rotated
            if page.rotation:
                center = (cim.shape[1] / 2, cim.shape[0] / 2)
                bbox = [rotate_point(p, -page.rotation, center) for p in bbox]

            def position_convert(x, cim_shape, page_rect):
                return (
                    float(x[0]) / cim_shape[1] * page_rect.width,
                    float(x[1]) / cim_shape[0] * page_rect.height,
                )

            R = fitz.Rect(
                position_convert(bbox[0], cim.shape, page.rect),
                position_convert(bbox[2], cim.shape, page.rect),
            )
            fn = "helv" if word.isascii() else "china-s"
            fs = R.width / fitz.get_text_length(word, fn, 1)
            page.insert_text(
                (R.x0, R.y1),
                word,
                fontname=fn,
                fontsize=fs,
                render_mode=3,
                rotate=page.rotation,
            )
            if args.pure:
                new_page.insert_text(
                    (R.x0, R.y1),
                    word,
                    fontname=fn,
                    fontsize=fs,
                    render_mode=0,
                    rotate=page.rotation,
                )
    if args.cv:
        cv2.destroyAllWindows()
    if args.pure:
        if pdf_doc.get_page_labels():
            pure.set_page_labels(pdf_doc.get_page_labels())
        if pdf_doc.get_toc():
            pure.set_toc(pdf_doc.get_toc())
        pure.save(
            pathlib.Path(output_pdf_path).stem + "-pure.pdf", garbage=4, deflate=True
        )
        pure.close()

    pdf_doc.save(output_pdf_path, garbage=4, deflate=True)
    pdf_doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program that adds hidden(but copiable) text layer to image pdf.",
        epilog="Copyright (C) 2024 Cao Yang. This is free software; distributed under GPLv3. There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.",
    )
    parser.add_argument("input_file", help="Input PDF file")
    parser.add_argument("output_file", help="Output PDF file")
    parser.add_argument("-p", "--pure", action="store_true")
    parser.add_argument("-c", "--cv", action="store_true")
    parser.add_argument("-n", "--no-ocr", action="store_true")
    parser.add_argument("-l", "--lang", default="en")

    args = parser.parse_args()
    if args.cv:
        cv2.namedWindow(sys.argv[0], cv2.WINDOW_NORMAL)
    process_pdf(args.input_file, args.output_file)
