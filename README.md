
# CorpusAid

CorpusAid is an advanced, user-friendly software tool designed specifically for preprocessing files in corpora compilation. This powerful application stands out for its ability to apply both personalized and traditional cleaning parameters across an entire corpus, regardless of its size. Whether you're working with a small collection of 10 files or a massive dataset of 10,000 documents, CorpusAid ensures consistent and accurate preprocessing.






[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)





## Features

CorpusAid offers a robust set of features designed to streamline and enhance the corpus preprocessing workflow:

### 1. Comprehensive Cleaning Parameters

CorpusAid provides a rich array of text cleaning options, giving users fine-grained control over their preprocessing tasks:

#### Text Normalization

- **Lowercase Conversion:** Standardizes text by converting all characters to lowercase, ensuring consistency in text analysis.
- **Whitespace Normalization:** Ensures uniform spacing throughout the text by removing redundant spaces, tabs, and other whitespace characters.
- **Line Break Removal:** Merges multiple lines to create continuous text, useful for certain types of analysis that require unbroken text streams.
- **Unicode Normalization:** Converts text into a standard Unicode format (e.g., NFC, NFD, NFKC, NFKD), ensuring consistency in character representation across different languages and scripts.

#### Character and Symbol Management

- **Punctuation Removal:** Strips away punctuation marks to focus on core textual content.
- **Number Removal:** Eliminates numerical digits when they're not relevant to the analysis.
- **Special Characters Removal:** Removes symbols and characters that may interfere with text processing algorithms.
- **Diacritic Removal:** Strips accents and diacritical marks from characters, useful for certain types of cross-linguistic analysis.
- **Greek and Cyrillic Character Removal:** Selectively filters out specific scripts as required by the research parameters.

#### Content Extraction and Cleaning

- **HTML Tag Stripping:** Removes HTML tags to extract plain text content from web-scraped or marked-up documents.
- **Bibliographical Reference Removal:** Automatically identifies and removes in-text bibliographical references (e.g., citations like `(Smith, 2020)`), cleaning up the text for analysis.
- **Custom Regular Expression Filtering:** Allows users to define and apply custom patterns for advanced text filtering and extraction.

#### Linguistic Processing

- **Lemmatization:** Reduces words to their base or dictionary form (lemmas), aiding in linguistic analysis by grouping together different forms of a word.
- **Sentence Tokenization:** Splits text into individual sentences, fundamental for tasks like sentiment analysis and syntactic parsing.
- **Word Tokenization:** Divides text into individual words or tokens, essential for word-level analysis and processing.
- **Stop Word Removal:** Excludes common, non-informative words (e.g., "the", "and") to focus on meaningful content.

### 2. Batch Processing

- Process entire directories of text files simultaneously, regardless of the number of files.
- Apply consistent preprocessing across large corpora, ensuring uniformity in your dataset.
- Dramatically reduce processing time compared to manual or file-by-file approaches.

### 3. Intuitive Graphical User Interface (GUI)

- User-friendly interface designed for researchers of all technical levels.
- Easy-to-navigate controls for selecting and applying preprocessing parameters.
- Real-time preview feature to see the effects of selected parameters on sample text.
- Progress indicators for tracking batch processing operations.

### 4. Detailed Summary Reporting

After processing, CorpusAid generates a comprehensive summary report that provides invaluable insights into your corpus:

- Word frequency distributions
- Sentence and token counts
- Type-token ratio analysis
- Corpus size statistics (pre and post-processing)
- Applied preprocessing parameters summary
- Processing time and performance metrics

### 5. Customization and Flexibility

- Save and load preprocessing profiles for consistent application across projects.
- Adjustable parameters to fine-tune preprocessing for specific research needs.
- Support for multiple input and output file formats (e.g., .txt, .csv, .json).

### 6. Data Integrity and Security

- Non-destructive processing: always keeps original files intact.
- Option to create backups automatically before processing.
- Detailed logging for audit trails and reproducibility.

### 7. Scalability

- Efficiently handles corpora of all sizes, from small datasets to large-scale collections.
- Optimized for performance on both personal computers and high-performance computing environments.

### 8. Interoperability

- Export preprocessed data in formats compatible with popular corpus analysis tools.
- Integration capabilities with other NLP pipelines and workflows.



## Screenshots

![App Screenshot](https://photos.fife.usercontent.google.com/pw/AP1GczNyi3WgVF5u-6QUwLpbPwR4cHdynv71ARNgKXdmhaBygbSMdLVAf86ypQ=w1643-h869-s-no-gm?authuser=0)

## Installation

CorpusAid is distributed as a standalone executable installer for Windows.

1. Download the `corpusaid_win_setup` file from <https://github.com/jhlopesalves/CorpusAid/releases>.
2. Run the installer and follow the on-screen instructions.
3. Once the installation is complete, you can launch CorpusAid from your desktop or Start Menu.

