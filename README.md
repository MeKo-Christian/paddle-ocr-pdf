# paddle-ocr-pdf

A Python script that uses PaddleOCR to add a hidden, selectable text layer to image-based PDFs.

Compared with OCRmyPDF, it does not insert random spaces between Chinese characters.

## Installation

### Install `paddlepaddle`

See [https://paddlepaddle.github.io/PaddleOCR/latest/quick\_start.html](https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html) for details, especially if you have a GPU.

```bash
python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

### Install `paddleocr`

```bash
pip install paddleocr
```

### Install `PyMuPDF`

```bash
pip install PyMuPDF
```

## Usage

### Basic usage

All three scripts—`pocr.py`, `pocr-pixmap.py`, and `pocr-inplace.py`—can add a hidden, copyable text layer to image PDFs. The general command format is:

```bash
python <script_name> input.pdf output.pdf
```

Replace `script_name` with the actual script name. `input.pdf` is your image-based PDF; `output.pdf` is the processed file.

### Command-line options

* **`-p` or `--pure`**: Generate a text-only PDF that contains only the OCR text layer. The output filename ends with `-pure.pdf`.

  ```bash
  python script_name.py -p input.pdf output.pdf
  ```

* **`-c` or `--cv`**: Display extracted images during processing.

  ```bash
  python script_name.py -c input.pdf output.pdf
  ```

* **`-n` or `--no-ocr`** (only for `pocr-pixmap.py` and `pocr-inplace.py`): Skip the OCR step.

  ```bash
  python script_name.py -n input.pdf output.pdf
  ```

* **`-l` or `--lang`**: Specify OCR language (defaults to `ch`).

  ```bash
  python script_name.py -l eng input.pdf output.pdf
  ```
