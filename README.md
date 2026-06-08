# Document Refinery Pipeline

This tool takes a PDF document, extracts the data inside, and organizes it into structured tables. It is designed to be reliable: it checks its own work, retries when something goes wrong, and learns from past runs to improve over time.


# Potential Use Cases

- Turning unstructured data within documents into structured, tabular data.
- Feature Generation across unstructured data for use in models that work best with Tabular data (e.g CATBoost)
- Synthetic data generation for Fine-Tuning tasks. 

## What it does

An agent acts as an assistant that: 

1. **Reads** the PDF and splits it into pages.
2. **Groups** pages that talk about similar things (like all invoice pages together, all report pages together).
3. **Decides** what information to look for on each group (e.g., date, amount, description).
4. **Looks** at each page image and pulls out the requested data.
5. **Checks** its own work by comparing the extracted data to the original text and by running consistency tests.
6. **Saves** everything into a single file (`master_output.json`) you can later load into a database or spreadsheet.

If a page is hard to read, the assistant tries again up to three times. If the whole process fails for one group, it continues with the others and tells you what went wrong.

## How it works (step by step)

1. **Ingestion**  
   The PDF is opened and each page is read. Pages with very little text (fewer than 15 characters) are skipped.

2. **Understanding the content**  
   The text of every page is converted into a mathematical representation called an "embedding" (a list of numbers that captures the meaning). Pages with similar meaning are placed close together in a high-dimensional space.

3. **Grouping pages (clustering)**  
   The pipeline finds dense groups of similar pages using an algorithm called HDBSCAN. Each group becomes a "cluster". Pages that don't belong to any group are marked as noise and ignored.

4. **Deciding what to extract (schema discovery)**  
   For each cluster, the pipeline reads a few representative pages and asks a language model: "What structured information could you find here?" The model suggests a list of fields (e.g., `item_description`, `price`, `quantity`). If the suggestion is invalid, the pipeline tries again up to four times.  
   If you run the pipeline multiple times on similar documents, it remembers which schemas produced the best results and reuses them (like genetic improvement).

5. **Extracting the data (vision model)**  
   Each page of the cluster is converted into an image. The image and the suggested fields are sent to a vision-language model (Qwen2.5-VL) that "looks" at the page and returns the data in the correct format.  
   The model is forced to produce valid JSON that matches the expected structure, avoiding garbled output.

6. **Verification**  
   The extracted data is checked in three ways:
   - **Semantic check:** A second model (Cross-Encoder) checks if the extracted values actually appear in or match the page text.
   - **Structure check:** The pipeline verifies that all expected fields are present and not empty.
   - **SQL check:** A small database engine (DuckDB) tries to store the data; if it fails, the extraction is considered invalid.

   A score (0–100) is calculated. If the score is below a target (default 85), the pipeline retries the page. After multiple retries, it picks the best result.

7. **Fact verification (optional)**  
   If enabled, each extracted row is turned into a question (e.g., "How much was spent on Office Supplies?") and the pipeline looks for the answer in the original text. Only rows that are confirmed by the source text are marked as verified.

8. **Cross-cluster reconciliation**  
   If multiple clusters contain similar data (e.g., invoices and purchase orders), the pipeline attempts to cross-reference them using SQL joins to catch mismatches (like missing items or price differences).

9. **Output**  
   All extracted data is saved to `master_output.json`. The file also includes a summary of scores, any errors that occurred, and statistics about each cluster.

## Requirements

- Python 3.11 or newer
- Ollama running locally with the model `qwen2.5-vl:7b` (or a compatible vision model; support for APIs and non-local models coming soon!)
- The following Python packages (install with `pip install -r requirements.txt`):
