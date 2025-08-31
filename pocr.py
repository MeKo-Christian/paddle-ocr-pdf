#!/usr/bin/env python3
from paddleocr import PaddleOCR
import fitz
import cv2
import numpy as np
import tqdm
import sys
import argparse
import pathlib

def process_pdf(input_pdf_path, output_pdf_path):
    pdf_doc = fitz.open(input_pdf_path)
    img = fitz.open()
    pure = fitz.open() if args.pure else None

    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang=args.lang,
    )

    for page_number in tqdm.tqdm(range(pdf_doc.page_count)):
        page = pdf_doc.load_page(page_number)

        # Remove existing selectable text; keep images
        page.add_redact_annot(page.rect)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # Decide render scale:
        # If there is an image on the page, render the page to that image's resolution
        # (largest one) so OCR pixels match the background pixels 1:1.
        images = page.get_images(full=True)
        if images:
            # pick the largest image by pixel area
            iw = ih = 0
            for im in images:
                w, h = im[2], im[3]
                if (w * h) > (iw * ih):
                    iw, ih = w, h
            # Fallback if metadata is weird
            if iw <= 0 or ih <= 0:
                zoom_x = zoom_y = 3.0
            else:
                zoom_x = iw / page.rect.width
                zoom_y = ih / page.rect.height
        else:
            # No images? Still rasterize the page at a decent DPI
            zoom_x = zoom_y = 3.0  # ~216 DPI

        # Render the page exactly as displayed (rotation/CTM baked in)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom_x, zoom_y), alpha=False)
        # Convert to OpenCV BGR for OCR
        np_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        cim = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

        if args.cv:
            cv2.imshow(sys.argv[0], cim)
            cv2.waitKey(1)

        # New output page uses displayed page size; background is the rendered pixmap
        out_page = img.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, pixmap=pix)  # NO extra rotation

        # Optional "pure" text-only PDF in pixel space
        if args.pure:
            pure_page = pure.new_page(width=cim.shape[1], height=cim.shape[0])

        # OCR
        pred = ocr.predict(cim)
        result = pred[0] if isinstance(pred, list) and pred else pred
        if not result or not result.get('rec_texts'):
            continue

        # Map pixel coords -> page coords (same orientation because we rendered the page)
        def px_to_page(pt):
            return (
                float(pt[0]) / cim.shape[1] * page.rect.width,
                float(pt[1]) / cim.shape[0] * page.rect.height,
            )

        for word, bbox, conf in zip(result['rec_texts'], result['rec_polys'], result['rec_scores']):
            if conf < 0.9:
                continue

            # bbox: 4 points in pixel coords (as rendered)
            x0, y0 = px_to_page(bbox[0])
            x1, y1 = px_to_page(bbox[2])
            R = fitz.Rect(x0, y0, x1, y1)

            fn = "helv" if word.isascii() else "china-s"
            tl = fitz.get_text_length(word, fn, 1)
            fs = (R.width / tl) if tl > 0 else 1.0

            # Insert invisible text aligned with rendered background (no extra rotation!)
            out_page.insert_text((R.x0, R.y1), word, fontname=fn, fontsize=fs, render_mode=3)

            if args.pure:
                # Text-only PDF: place in pixel coordinates directly
                px0 = min(p[0] for p in bbox); py0 = min(p[1] for p in bbox)
                px1 = max(p[0] for p in bbox); py1 = max(p[1] for p in bbox)
                Rp = fitz.Rect(px0, py0, px1, py1)
                tlp = fitz.get_text_length(word, fn, 1)
                fsp = (Rp.width / tlp) if tlp > 0 else 1.0
                pure_page.insert_text((Rp.x0, Rp.y1), word, fontname=fn, fontsize=fsp, render_mode=0)

    if args.cv:
        cv2.destroyAllWindows()

    # Preserve labels / TOC
    if args.pure:
        if pdf_doc.get_page_labels():
            pure.set_page_labels(pdf_doc.get_page_labels())
        if pdf_doc.get_toc():
            pure.set_toc(pdf_doc.get_toc())
        pure.save(pathlib.Path(output_pdf_path).stem + "-pure.pdf", garbage=4, deflate=True)
        pure.close()

    if pdf_doc.get_page_labels():
        img.set_page_labels(pdf_doc.get_page_labels())
    if pdf_doc.get_toc():
        img.set_toc(pdf_doc.get_toc())

    img.save(output_pdf_path, garbage=4, deflate=True)
    img.close()
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
    parser.add_argument("-l", "--lang", default="ch")

    args = parser.parse_args()
    if args.cv:
        cv2.namedWindow(sys.argv[0], cv2.WINDOW_NORMAL)
    process_pdf(args.input_file, args.output_file)
