# What is CorpuScript?

CorpuScript is an advanced, user-friendly software tool designed specifically for preprocessing files in corpora compilation. This powerful application stands out for its ability to apply both personalized and traditional cleaning parameters across an entire corpus, regardless of its size. Whether you're working with a small collection of 10 files or a massive dataset of 10,000 documents, CorpuScript ensures consistent and accurate preprocessing.

## Key features of CorpuScript include

1. **Customizable preprocessing parameters**: Tailor the cleaning process to your specific research needs.
2. **Batch processing**: Apply selected parameters to all files in your corpus simultaneously.
3. **Scalability**: Efficiently handle corpora of varying sizes, from small datasets to large-scale collections.
4. **Consistency**: Eliminate human error by automating the preprocessing steps across all documents.
5. **Time-saving**: Dramatically reduce the time required for corpus preparation, potentially cutting months or even years off your research timeline.
6. **User-friendly interface**: Designed to be accessible for users with varying levels of technical expertise.

## Purpose and design goals

The primary purpose of CorpuScript is to streamline and standardize the often tedious and error-prone process of preparing textual data for corpus linguistics research. Its design goals include:

1. **Efficiency**: Automate repetitive tasks to save researchers valuable time and resources.
2. **Accuracy**: Minimize human error in the preprocessing stage, ensuring more reliable research outcomes.
3. **Flexibility**: Provide a wide range of preprocessing options to suit various research methodologies and corpus types.
4. **Accessibility**: Create a tool that can be used effectively by researchers at all levels, from students to seasoned professionals.
5. **Reproducibility**: Enable consistent application of preprocessing parameters, enhancing the reproducibility of corpus-based studies.

## Who is it for?

CorpuScript is an invaluable tool for a wide range of users in the field of linguistics and beyond:

- **Students**: Undergraduate and graduate students working on corpus-based projects or theses.
- **Professors**: Academic staff preparing corpora for research projects or teaching materials.
- **Researchers**: Linguistics researchers, computational linguists, and natural language processing specialists.
- **Language professionals**: Translators, lexicographers, and language teachers.
- **Data scientists**: Those working with text-based data in social sciences, digital humanities, or market research.
- **Corpus linguists**: Professionals specializing in corpus linguistics.
- **Natural Language Processing (NLP) practitioners**: Those developing language models or conducting text analysis.

## Features

CorpuScript offers a robust set of features designed to streamline and enhance the corpus preprocessing workflow:

### 1. Comprehensive Cleaning Parameters

CorpuScript provides a rich array of text cleaning options, giving users fine-grained control over their preprocessing tasks:

#### Text Normalization

- **Lowercase Conversion:** Standardizes text by converting all characters to lowercase, ensuring consistency in text analysis.
- **Whitespace Normalization:** Ensures uniform spacing throughout the text by removing redundant spaces, tabs, and other whitespace characters.
- **Line Break Removal:** Merges multiple lines to create continuous text, useful for certain types of analysis that require unbroken text streams.

#### Character and Symbol Management

- **Punctuation Removal:** Strips away punctuation marks to focus on core textual content.
- **Number Removal:** Eliminates numerical digits when they're not relevant to the analysis.
- **Special Characters Removal:** Removes symbols and characters that may interfere with text processing algorithms.
- **Diacritic Removal:** Strips accents and diacritical marks from characters, useful for certain types of cross-linguistic analysis.
- **Greek and Cyrillic Character Removal:** Selectively filters out specific scripts as required by the research parameters.

#### Content Extraction and Cleaning

- **HTML Tag Stripping:** Removes HTML tags to extract plain text content from web-scraped or marked-up documents.
- **Custom Regular Expression Filtering:** Allows users to define and apply custom patterns for advanced text filtering and extraction.

#### Linguistic Processing

- **Lemmatization:** Reduces words to their base forms (lemmas) to facilitate more accurate linguistic analysis.
- **Tokenization:** Splits text into individual words or tokens, a fundamental step for many NLP tasks.
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

After processing, CorpuScript generates a comprehensive summary report that provides invaluable insights into your corpus:

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

### 9. Multilingual Support

- Built-in support for processing texts in multiple languages.
- Customizable language-specific preprocessing rules.

### 10. Continuous Updates and Community Support

- Regular updates to incorporate the latest advancements in corpus linguistics and NLP.
- Active user community for sharing best practices and custom preprocessing recipes.

## Installation

PreTextCleaner is distributed as a standalone executable installer for Windows.

1. Download the `setup.exe` file from [link to your distribution].
2. Run the installer and follow the on-screen instructions.
3. Once the installation is complete, you can launch PreTextCleaner from your desktop or Start Menu.

## Usage

PreTextCleaner is designed to be easy to use. Here's a typical workflow:

1. **Load Text Files:** Open individual text files or entire directories using the "Open Files" or "Open Directory" options from the File menu.
2. **Select Processing Parameters:** Customize your text cleaning process by selecting the desired parameters in the "Processing Parameters" dialog. Options include setting filters, choosing which characters to remove, and configuring advanced text processing modules like lemmatization and tokenization.
3. **Process Files:** Click the "Process Files" button to start cleaning and preprocessing your text data. You can monitor the progress in real-time through the progress bar and status updates.
4. **View Results:** Once processing is complete, view the cleaned text in the "Processed Text" tab. You can also generate and view a detailed summary report in the "Summary Report" tab. Export the processed text or summary report as needed.

## Example Use Cases

- **Corpus Linguistics Research:** Clean and prepare text corpora for linguistic analysis, enabling you to focus on your research questions.
- **Preprocessing for Other Tools:**  Prepare text data for use with other corpus analysis tools like Sketch Engine and Biber tagger.
- **General Text Cleaning:**  Use PreTextCleaner to clean and standardize text data for various NLP tasks and applications.

## License

PreTextCleaner is licensed under the MIT License. This means you are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that you include the following copyright notice in all copies or substantial portions of the software:

MIT License

Copyright (c) 2024 Jhonatan Henrique Lopes Alves

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact

<jhlopesalves@gmail.com>
