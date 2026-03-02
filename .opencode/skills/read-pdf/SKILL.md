---
name: read-pdf
description: Extract text from a PDF file using the pdftotext command-line utility.
---
## What I do
I provide instructions to extract all text from a PDF document using the `pdftotext` command-line utility.

## When to use me
Use me when you need to read the text content of a PDF file from the local filesystem.

## How to use me
1.  You will be given a path to a PDF file.
2.  First, check if `pdftotext` is installed by running `which pdftotext`.
3.  If the command is not found, you must inform the user that the `poppler-utils` package needs to be installed. Provide one of the following commands based on the user's OS:
    -   For Debian/Ubuntu: `sudo apt-get install poppler-utils`
    -   For macOS (with Homebrew): `brew install poppler`
    -   For other systems, ask the user to install `poppler-utils` or `pdftotext`.
    Do not proceed until the tool is installed.
4.  Once `pdftotext` is available, run the following command to extract the text from the PDF file. Replace `<path/to/your/file.pdf>` with the actual file path.

    ```bash
    pdftotext -layout <path/to/your/file.pdf> -
    ```

    - The `-layout` flag helps maintain the original layout of the text.
    - The final `-` tells `pdftotext` to print the output to standard output.
