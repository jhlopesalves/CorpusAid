import os
import sys
import re
import unicodedata
from charset_normalizer import from_bytes
import logging
import json
import urllib.request
import time
import random
import multiprocessing
from threading import Lock
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QSizePolicy,
    QTextEdit, QProgressBar, QFileDialog, QLabel, QListWidget, QToolButton, QStyle,
    QTabWidget, QLineEdit, QDialog, QDialogButtonBox, QCheckBox, QMessageBox,
    QSplitter, QToolBar, QStatusBar, QListWidgetItem,
    QPlainTextEdit, QWizard, QWizardPage, QTableWidget, QComboBox, QHeaderView, QStackedWidget, QGridLayout
)
from PySide6.QtCore import Qt, Signal, QObject, QThread, QSize, QRunnable, QThreadPool, QUrl, QTimer, QRegularExpression
from PySide6.QtGui import (
    QIcon, QFont, QColor, QAction, QPainter, QIntValidator, QTextCursor, QTextCharFormat, QSyntaxHighlighter, QDesktopServices, QKeySequence, QFontDatabase
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import threading

# resource_path function to get the path of the resource file
def resource_path(relative_path): 
    if getattr(sys, 'frozen', False):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# Function to get the spacy model
def get_spacy_model(): 
    if not hasattr(get_spacy_model, 'lock'):
        get_spacy_model.lock = threading.Lock()
    with get_spacy_model.lock:
        if not hasattr(get_spacy_model, 'nlp'):
            get_spacy_model.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            get_spacy_model.nlp.add_pipe('sentencizer')
    return get_spacy_model.nlp

# Function to load the fonts
def load_fonts():
    font_db = QFontDatabase()
    roboto_font_path = resource_path("fonts/Roboto-Regular.ttf")
    if os.path.exists(roboto_font_path):
        font_id = font_db.addApplicationFont(roboto_font_path)
        if font_id != -1:
            font_families = font_db.applicationFontFamilies(font_id)
            if font_families:
                QFont("Roboto")
    else:
        logging.warning("Roboto font not found. Falling back to default font.")

class PreprocessingModule:
    def process(self, text):
        raise NotImplementedError

class CharacterFilterModule(PreprocessingModule):
    def __init__(self, chars_to_remove):
        self.chars_to_remove = sorted(chars_to_remove, key=len, reverse=True)

    def process(self, text):
        for item in self.chars_to_remove:
            text = text.replace(item, '')
        return text

class PageNumberRemovalModule(PreprocessingModule):
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(r'(?:\n\s*){1,2}\d{1,4}(?:\s*\n){1,2}', re.MULTILINE)
    
    def process(self, text):
        return self.pattern.sub('\n', text)

class WhitespaceNormalizationModule(PreprocessingModule):
    def process(self, text):
        text = re.sub(r'\s+([.,?!;:])', r'\1', text) # Remove whitespace before punctuation
        text = re.sub(r'([.,?!;:])(\S)', r'\1 \2', text) # Add whitespace after punctuation
        text = re.sub(r'\(\s+', '(', text) # Remove whitespace after opening parenthesis
        text = re.sub(r'\s+\)', ')', text) # Remove whitespace before closing parenthesis
        text = re.sub(r'\[\s+', '[', text) # Remove whitespace after opening bracket
        text = re.sub(r'\s+\]', ']', text) # Remove whitespace before closing bracket
        text = re.sub(r'\{\s+', '{', text) # Remove whitespace after opening brace
        text = re.sub(r'\s+\}', '}', text) # Remove whitespace before closing brace
        text = re.sub(r'\s{2,}', ' ', text) # Replace multiple spaces with a single space
        return text.strip()

class LineBreakRemovalModule(PreprocessingModule):
    def process(self, text):
        return text.replace('\n', ' ')

class BibliographicalReferenceRemovalModule(PreprocessingModule):
    def __init__(self):
        super().__init__()
        # Updated regex pattern to match bibliographical references more precisely
        self.pattern = re.compile(r'\([A-Z][a-z]+(?:[^()]*?\d{4}[^()]*?)?\)')
    
    def process(self, text: str) -> str:
        return self.pattern.sub('', text)

class LowercaseModule(PreprocessingModule):
    def process(self, text):
        return text.lower()

class TokenBasedModule(PreprocessingModule):
    def __init__(self):
        self.nlp = get_spacy_model()

    def process_tokens(self, tokens):
        raise NotImplementedError

    def process(self, text_or_tokens):
        if isinstance(text_or_tokens, str):
            doc = self.nlp(text_or_tokens)
            tokens = [token.text for token in doc]
        else:
            tokens = text_or_tokens
        return self.process_tokens(tokens)

class RegexSubstitutionModule(PreprocessingModule): 
    def __init__(self, pattern, replacement=''):
        super().__init__()
        try:
            self.pattern = re.compile(pattern, re.DOTALL)
            logging.debug(f"Compiled regex pattern: {pattern}")
        except re.error as e:
            logging.error(f"Invalid regex pattern: {pattern}\nError: {str(e)}")
            QMessageBox.warning(None, "Invalid Regex Pattern", f"The entered regex pattern is invalid:\n{str(e)}")
            self.pattern = None
        self.replacement = replacement

    def process(self, text):
        if self.pattern:
            new_text, count = self.pattern.subn(self.replacement, text)
            if count > 0:
                logging.debug(f"Applied pattern: '{self.pattern.pattern}' - {count} replacements made.")
            return new_text
        return text

class WordTokenizationModule(TokenBasedModule):
    def process(self, text):
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        return ' '.join(tokens)

class StopWordRemovalModule(TokenBasedModule):
    def __init__(self):
        super().__init__()
        self.stop_words = set(STOP_WORDS)

    def process_tokens(self, tokens):
        return [word for word in tokens if word.lower() not in self.stop_words]

class HTMLStripperModule(PreprocessingModule):
    def process(self, text):
        return BeautifulSoup(text, "html.parser").get_text()

class DiacriticRemovalModule(PreprocessingModule):
    def process(self, text):
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

class GreekLetterRemovalModule(PreprocessingModule):
    def process(self, text):
        return ''.join(char for char in text if not unicodedata.name(char, '').startswith('GREEK'))

class CyrillicRemovalModule(PreprocessingModule):
    def process(self, text):
        return ''.join(char for char in text if not unicodedata.name(char, '').startswith('CYRILLIC'))

class UnicodeNormalizationModule(PreprocessingModule):
    def process(self, text):
        return unicodedata.normalize('NFKC', text)

class UnicodeCategoryFilterModule(PreprocessingModule):
    def __init__(self, categories_to_remove):
        self.categories_to_remove = set(categories_to_remove)

    def process(self, text):
        return ''.join(char for char in text if unicodedata.category(char) not in self.categories_to_remove)

class PreprocessingPipeline:
    def __init__(self):
        self.modules = []

    def add_module(self, module):
        self.modules.append(module)

    def process(self, text):
        for module in self.modules:
            text = module.process(text)
            if isinstance(text, list):
                text = ' '.join(text)
        return text.strip()

class DocumentProcessor:
    default_parameters = {
        "remove_break_lines": False,
        "lowercase": False,
        "chars_to_remove": [],
        "word_tokenization": False,
        "remove_stop_words": False,
        "regex_pattern": "",
        "strip_html": False,
        "remove_diacritics": False,
        "remove_greek": False,
        "remove_cyrillic": False,
        "remove_super_sub_script": False,
        "remove_page_numbers": False,  
        "remove_bibliographical_references": False,
        "normalize_spacing": False,
        "normalize_unicode": False,
    }

    def __init__(self):
        self.parameters = self.default_parameters.copy()
        self.update_pipeline()

    def set_parameters(self, parameters):
        try:
            if "regex_pattern" in parameters and parameters["regex_pattern"]:
                re.compile(parameters["regex_pattern"])
            if "chars_to_remove" in parameters:
                if not isinstance(parameters["chars_to_remove"], list):
                    raise ValueError("chars_to_remove must be a list of strings")
                for item in parameters["chars_to_remove"]:
                    if not isinstance(item, str):
                        raise ValueError("All items in chars_to_remove must be strings")
            self.parameters.update(parameters)
            self.update_pipeline()
            logging.debug(f"Updated parameters: {self.parameters}")
        except re.error as e:
            logging.error(f"Invalid regex pattern: {e}")
            QMessageBox.warning(None, "Invalid Regex Pattern", f"The entered regex pattern is invalid:\n{str(e)}")
        except ValueError as e:
            logging.error(f"Parameter validation error: {e}")
            QMessageBox.warning(None, "Parameter Error", str(e))

    def reset_parameters(self):
        self.parameters = self.default_parameters.copy()
        self.update_pipeline()

    def update_pipeline(self):
        self.pipeline = PreprocessingPipeline()
        if self.parameters["regex_pattern"]:
            self.pipeline.add_module(RegexSubstitutionModule(self.parameters["regex_pattern"]))
        if self.parameters["strip_html"]:
            self.pipeline.add_module(HTMLStripperModule())
        if self.parameters["remove_break_lines"]:
            self.pipeline.add_module(LineBreakRemovalModule())
        if self.parameters["lowercase"]:
            self.pipeline.add_module(LowercaseModule())
        if self.parameters["chars_to_remove"]:
            self.pipeline.add_module(CharacterFilterModule(self.parameters["chars_to_remove"]))
        if self.parameters["word_tokenization"]:
            self.pipeline.add_module(WordTokenizationModule())
        if self.parameters["remove_stop_words"]:
            self.pipeline.add_module(StopWordRemovalModule())
        if self.parameters["remove_diacritics"]:
            self.pipeline.add_module(DiacriticRemovalModule())
        if self.parameters["remove_greek"]:
            self.pipeline.add_module(GreekLetterRemovalModule())
        if self.parameters["remove_cyrillic"]:
            self.pipeline.add_module(CyrillicRemovalModule())
        if self.parameters["remove_super_sub_script"]:
            categories_to_remove = {'No', 'Sk'}
            self.pipeline.add_module(UnicodeCategoryFilterModule(categories_to_remove))
        if self.parameters["remove_page_numbers"]:
            self.pipeline.add_module(PageNumberRemovalModule())  
        if self.parameters["remove_bibliographical_references"]:
            self.pipeline.add_module(BibliographicalReferenceRemovalModule())
        if self.parameters["normalize_spacing"]:
            self.pipeline.add_module(WhitespaceNormalizationModule())
        if self.parameters["normalize_unicode"]:
            self.pipeline.add_module(UnicodeNormalizationModule())

    def get_parameters(self):
        return self.parameters

    def process_file(self, text):
        if not any(self.parameters.values()):
            return text
        processed_text = self.pipeline.process(text)
        logging.debug("Processed text through pipeline.")
        return processed_text
    
class ProcessingWorker(QRunnable):
    def __init__(self, processor, file_path, signals):
        super().__init__()
        self.processor = processor
        self.file_path = file_path
        self.signals = signals
        self.setAutoDelete(True)

    def run(self):
        try:
            file_size = os.path.getsize(self.file_path)
        except Exception as e:
            self.signals.error.emit(self.file_path, f"Failed to get file size: {str(e)}")
            return

        try:
            # Detect encoding
            with open(self.file_path, 'rb') as f:
                raw_data = f.read()
                result = from_bytes(raw_data)
                best_guess = result.best()
                if best_guess:
                    encoding = best_guess.encoding
                else:
                    encoding = 'utf-8'  # Fallback encoding

            start_time = time.time()
            with open(self.file_path, 'r', encoding=encoding, errors='replace') as file:
                original_text = file.read()

            processed_text = self.processor.process_file(original_text)
            end_time = time.time()
            processing_time = end_time - start_time

            self.signals.result.emit(self.file_path, original_text, processed_text, file_size, processing_time)
        except Exception as e:
            self.signals.error.emit(self.file_path, f"Unexpected error while reading file: {str(e)}")
            return
        finally:
            self.signals.finished.emit()

class ProcessingSignals(QObject):
    result = Signal(str, str, str, int, float) 
    error = Signal(str, str)
    warning = Signal(str, str)
    finished = Signal()
    update_progress = Signal(int, int, float, str)
    processing_complete = Signal(list, list)
    report_ready = Signal(str, str)

class FileManager:
    def __init__(self):
        self.files = []

    def add_files(self, file_paths): 
        new_files = [os.path.normpath(f) for f in file_paths if os.path.normpath(f) not in self.files]
        self.files.extend(new_files)
        return new_files

    def add_directory(self, directory, signals, is_cancelled_callback): 
        temp_files = []
        new_files = []
        total_files = 0
        
        for root, dirs, files in os.walk(directory):
            total_files += len([f for f in files if f.endswith(".txt")])
        start_time = time.time()
        processed_files = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if is_cancelled_callback():
                    signals.update_progress.emit(processed_files, total_files, 0, "Operation cancelled.")
                    return []
                
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    normalized_path = os.path.normpath(file_path)
                    if normalized_path not in self.files and normalized_path not in temp_files:
                        try:
                            time.sleep(random.uniform(0.01, 0.1))
                            temp_files.append(normalized_path)
                            new_files.append(normalized_path)
                            processed_files += 1
                            elapsed_time = time.time() - start_time
                            estimated_remaining_time = (elapsed_time / processed_files) * (total_files - processed_files) if processed_files > 0 else 0
                            signals.update_progress.emit(processed_files, total_files, estimated_remaining_time, None)
                        except OSError as e:
                            signals.update_progress.emit(processed_files, total_files, 0, f"OS error: {str(e)}")
                        except Exception as e:
                            signals.update_progress.emit(processed_files, total_files, 0, f"Unexpected error: {str(e)}")
        self.files.extend(temp_files)
        return new_files

    def remove_files(self, file_paths):
        for file in file_paths:
            if file in self.files:
                self.files.remove(file)

    def clear_files(self):
        self.files.clear()

    def get_files(self):
        return self.files

    def get_total_size(self):
        return sum(os.path.getsize(file) for file in self.files)

class FileListWidget(QListWidget):
    files_added = Signal(list)
    files_removed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setDragDropMode(QListWidget.InternalMove)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                file_path = os.path.normpath(url.toLocalFile())
                links.append(file_path)
            self.addItems(links)
            self.files_added.emit(links)
        else:
            super().dropEvent(event)
            source_row = self.currentRow()
            destination_row = self.row(self.itemAt(event.pos()))
            if source_row != destination_row:
                item = self.takeItem(source_row)
                self.insertItem(destination_row, item)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            selected_items = self.selectedItems()
            for item in selected_items:
                self.takeItem(self.row(item))
            self.files_removed.emit([item.text() for item in selected_items])
        else:
            super().keyPressEvent(event)

class ThemeManager:
    def __init__(self):
        self.dark_theme = True
        self.custom_colors = {
            "primary": "#518FBC",
            "secondary": "#325F84",
            "background": "#1E1E1E",
            "text": "#FFFFFF",
            "accent": "#FFB900",
            "icon_dark": "#FFFFFF",
            "icon_light": "#000000",
            "border": "#3F3F3F",
            "widget_background": "#2D2D2D",
            "report_background": "#2D2D2D",
            "report_text": "#FFFFFF",
            "report_description": "#A0A0A0"
        }

    def toggle_theme(self):
        self.dark_theme = not self.dark_theme
        if self.dark_theme:
            self.custom_colors.update({
                "background": "#1E1E1E",
                "text": "#FFFFFF",
                "widget_background": "#2D2D2D",
                "report_background": "#2D2D2D",
                "border": "#3F3F3F",
                "report_text": "#FFFFFF",
                "report_description": "#A0A0A0"
            })
        else:
            self.custom_colors.update({
                "background": "#F0F0F0",
                "text": "#000000",
                "widget_background": "#FFFFFF",
                "report_background": "#FFFFFF",
                "border": "#CCCCCC",
                "report_text": "#000000",
                "report_description": "#505050"
            })

    def get_stylesheet(self):
        return f"""
            QMainWindow, QWidget {{
                background-color: {self.custom_colors['background']};
                color: {self.custom_colors['text']};
            }}
            QPushButton {{
                background-color: {self.custom_colors['primary']};
                color: {self.custom_colors['text']};
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {self.custom_colors['secondary']};
            }}
            QTextEdit, QListWidget, QPlainTextEdit, QWidget#reportWidget {{
                border: 1px solid {self.custom_colors['border']};
                background-color: {self.custom_colors['widget_background']};
                color: {self.custom_colors['text']};
                border-radius: 5px;
                padding: 5px;
            }}
            QLabel#sectionLabel {{
                color: {self.custom_colors['primary']};
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            QLabel#reportValue {{
                color: {self.custom_colors['report_text']};
                font-size: 24px;
                font-weight: bold;
            }}
            QLabel#reportDescription {{
                color: {self.custom_colors['report_description']};
                font-size: 12px;
            }}
            QWidget#reportWidget {{
                background-color: {self.custom_colors['report_background']};
            }}
            QTabWidget::pane {{
                border: 1px solid {self.custom_colors['border']};
                border-radius: 5px;
            }}
            QTabBar::tab {{
                background-color: {self.custom_colors['widget_background']};
                color: {self.custom_colors['text']};
                padding: 5px 10px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.custom_colors['primary']};
                color: {self.custom_colors['text']};
            }}
            QLineEdit {{
                border: 1px solid {self.custom_colors['border']};
                background-color: {self.custom_colors['widget_background']};
                color: {self.custom_colors['text']};
                padding: 3px;
                border-radius: 3px;
            }}
            QMenuBar {{
                background-color: {self.custom_colors['background']};
                color: {self.custom_colors['text']};
            }}
            QMenuBar::item {{
                background-color: transparent;
            }}
            QMenuBar::item:selected {{
                background-color: {self.custom_colors['secondary']};
            }}
            QMenuBar::item:pressed {{
                background-color: {self.custom_colors['primary']};
            }}
            QMenu {{
                background-color: {self.custom_colors['background']};
                color: {self.custom_colors['text']};
                border: 1px solid {self.custom_colors['border']};
            }}
            QMenu::item:selected {{
                background-color: {self.custom_colors['secondary']};
            }}
        """
    
    def update_color(self, color_key, color_value):
        if color_key in self.custom_colors:
            self.custom_colors[color_key] = color_value

class AdvancedPatternBuilder(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Pattern Builder")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setMinimumSize(700, 500)
        self.addPage(self.createPatternPage())
        self.addPage(self.createPreviewPage())

    def createPatternPage(self):
        page = QWizardPage()
        page.setTitle("Define Patterns")
        layout = QVBoxLayout()
        self.pattern_table = QTableWidget()
        self.pattern_table.setColumnCount(4)
        self.pattern_table.setHorizontalHeaderLabels(["Start Condition", "End Condition Type", "End Condition", "Number Length"])
        self.pattern_table.horizontalHeader().setStretchLastSection(False)
        self.pattern_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        header = self.pattern_table.horizontalHeader()
        for i in range(4):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        layout.addWidget(self.pattern_table)
        add_button = QPushButton("Create Pattern")
        add_button.clicked.connect(self.addPattern)
        layout.addWidget(add_button)
        options_layout = QHBoxLayout()
        self.case_sensitive = QCheckBox("Case sensitive")
        options_layout.addWidget(self.case_sensitive)
        self.whole_words = QCheckBox("Match whole words only")
        options_layout.addWidget(self.whole_words)
        options_layout.addStretch()
        layout.addLayout(options_layout)
        page.setLayout(layout)
        return page

    def createPreviewPage(self):
        page = QWizardPage()
        page.setTitle("Preview and Test")
        layout = QVBoxLayout()
        self.pattern_preview = QPlainTextEdit()
        self.pattern_preview.setReadOnly(True)
        layout.addWidget(QLabel("Pattern Preview:"))
        layout.addWidget(self.pattern_preview)
        self.explanation = QLabel()
        self.explanation.setWordWrap(True)
        layout.addWidget(QLabel("Explanation:"))
        layout.addWidget(self.explanation)
        self.test_input = QTextEdit()
        self.test_input.setPlaceholderText("Enter test text here")
        layout.addWidget(QLabel("Test your pattern:"))
        layout.addWidget(self.test_input)
        self.test_button = QPushButton("Test Pattern")
        self.test_button.clicked.connect(self.testPattern)
        layout.addWidget(self.test_button)
        page.setLayout(layout)
        return page

    def addPattern(self):
        row_position = self.pattern_table.rowCount()
        self.pattern_table.insertRow(row_position)
        start_edit = QLineEdit()
        self.pattern_table.setCellWidget(row_position, 0, start_edit)
        end_type_combo = QComboBox()
        end_type_combo.addItems(["Single Number", "Multiple Numbers", "Specific Sequence"])
        self.pattern_table.setCellWidget(row_position, 1, end_type_combo)
        end_type_combo.currentIndexChanged.connect(lambda index, row=row_position: self.updateEndCondition(row, index))
        end_edit = QLineEdit()
        self.pattern_table.setCellWidget(row_position, 2, end_edit)
        number_length_edit = QLineEdit()
        number_length_validator = QIntValidator(1, 100, self)
        number_length_edit.setValidator(number_length_validator)
        number_length_edit.setEnabled(False)
        self.pattern_table.setCellWidget(row_position, 3, number_length_edit)

    def updateEndCondition(self, row, index):
        end_edit = self.pattern_table.cellWidget(row, 2)
        number_length_edit = self.pattern_table.cellWidget(row, 3)
        if index == 0:  
            end_edit.setEnabled(True)
            end_edit.setValidator(QIntValidator(0, 9, self))
            number_length_edit.setEnabled(False)
        elif index == 1:  
            end_edit.setEnabled(False)
            number_length_edit.setEnabled(True)
        elif index == 2:  
            end_edit.setEnabled(True)
            end_edit.setValidator(None)
            number_length_edit.setEnabled(False)

    def getPatternData(self):
        pattern_data = []
        for row in range(self.pattern_table.rowCount()):
            start = self.pattern_table.cellWidget(row, 0).text().strip()
            end_type = self.pattern_table.cellWidget(row, 1).currentText()
            end = self.pattern_table.cellWidget(row, 2).text().strip()
            number_length = self.pattern_table.cellWidget(row, 3).text().strip()
            if start and end:
                pattern_data.append({
                    "start": start,
                    "end_type": end_type,
                    "end": end,
                    "number_length": number_length
                })
        return pattern_data

    def updatePattern(self):
        try:
            pattern_data = self.getPatternData()
            patterns = []
            for data in pattern_data:
                start = re.escape(data["start"])
                if data["end_type"] == "Single Number":
                    end = r'\d'
                    pattern = rf"{start}.*?{end}"
                elif data["end_type"] == "Multiple Numbers":
                    if not data["number_length"].isdigit():
                        raise ValueError("Number Length must be a positive integer for Multiple Numbers.")
                    end = r'\d{' + data["number_length"] + '}'
                    pattern = rf"{start}.*?{end}"
                else:  # Specific Word
                    end = re.escape(data["end"])
                    pattern = rf"{start}.*?{end}"
                patterns.append(pattern)
            final_pattern = '|'.join(patterns)
            if self.whole_words.isChecked():
                final_pattern = rf"\b({final_pattern})\b"
            flags = re.DOTALL | (0 if self.case_sensitive.isChecked() else re.IGNORECASE)
            self.final_pattern = re.compile(final_pattern, flags)
            self.pattern_preview.setPlainText(final_pattern)
            self.explanation.setText(f"This pattern will match: {', '.join(patterns)}")
            logging.debug(f"Final regex pattern: {final_pattern} with flags: {flags}")
        except re.error as e:
            QMessageBox.warning(self, "Invalid Pattern", f"The entered pattern is invalid:\n{str(e)}")
            logging.error(f"Invalid regex pattern: {str(e)}")
        except ValueError as ve:
            QMessageBox.warning(self, "Invalid Input", str(ve))
            logging.error(f"Invalid input for pattern: {str(ve)}")

    def testPattern(self):
        self.updatePattern()
        text = self.test_input.toPlainText()
        if not hasattr(self, 'final_pattern') or not self.final_pattern:
            QMessageBox.warning(self, "No Pattern", "Please define a valid pattern first.")
            return
        try:
            matches = list(self.final_pattern.finditer(text))
            cursor = self.test_input.textCursor()
            text_format = QTextCharFormat()
            text_format.setBackground(Qt.yellow)
            cursor.beginEditBlock()
            cursor.select(QTextCursor.Document)
            cursor.setCharFormat(QTextCharFormat()) 
            cursor.clearSelection()
            for match in matches:
                cursor.setPosition(match.start())
                cursor.setPosition(match.end(), QTextCursor.KeepAnchor)
                cursor.setCharFormat(text_format)
            cursor.endEditBlock()
            if not matches:
                QMessageBox.information(self, "No Matches", "The pattern did not match any text in the sample.")
            else:
                QMessageBox.information(self, "Matches Found", f"Found {len(matches)} matches.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred while testing the pattern:\n{str(e)}")
            logging.error(f"Error testing pattern: {str(e)}")

    def getPattern(self):
        self.updatePattern()
        return self.final_pattern
        
class RegexHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.formats = {
            'meta_char': QTextCharFormat(),
            'quantifier': QTextCharFormat(),
            'grouping': QTextCharFormat(),
            'character_class': QTextCharFormat(),
            'escaped_char': QTextCharFormat(),
            'literal': QTextCharFormat(),
        }
        self.formats['meta_char'].setForeground(QColor("#C586C0"))
        self.formats['quantifier'].setForeground(QColor("#D16969"))
        self.formats['grouping'].setForeground(QColor("#4EC9B0"))
        self.formats['character_class'].setForeground(QColor("#DCDCAA"))
        self.formats['escaped_char'].setForeground(QColor("#9CDCFE"))
        self.formats['literal'].setForeground(QColor("#FFFFFF"))

    def highlightBlock(self, text):
        index = 0
        while index < len(text):
            char = text[index]
            if char == '\\' and index + 1 < len(text):
                self.setFormat(index, 2, self.formats['escaped_char'])
                index += 2
                continue
            if char in '.^$|?*+(){}[]':
                if char in '(){}[]':
                    self.setFormat(index, 1, self.formats['grouping'])
                elif char in '?*+':
                    self.setFormat(index, 1, self.formats['quantifier'])
                else:
                    self.setFormat(index, 1, self.formats['meta_char'])
                index += 1
                continue
            self.setFormat(index, 1, self.formats['literal'])
            index += 1

class FileDisplayDialog(QDialog):
    def __init__(self, total_files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Number of Files to Display")
        self.total_files = total_files

        layout = QVBoxLayout(self)

        label = QLabel(f"Total files selected: {total_files}\nHow many files would you like to display?")
        layout.addWidget(label)

        self.input_field = QLineEdit()
        self.input_field.setValidator(QIntValidator(1, total_files))
        self.input_field.setPlaceholderText(f"Enter a number between 1 and {total_files}")
        self.input_field.setText("10")  # Default value
        layout.addWidget(self.input_field)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)

    def validate_and_accept(self):
        text = self.input_field.text()
        if text.isdigit():
            value = int(text)
            if 1 <= value <= self.total_files:
                self.accept()
                return
        QMessageBox.warning(self, "Invalid Input", f"Please enter a valid number between 1 and {self.total_files}.")

    def get_number_of_files(self):
        return int(self.input_field.text())

class ParametersDialog(QDialog):
    def __init__(self, current_parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Parameters")
        self.setMinimumSize(200, 400)
        layout = QVBoxLayout(self)
        self.parameters = current_parameters.copy()
        self.pattern_data = self.parameters.get("pattern_data", [])

        self.options = [
            ("remove_break_lines", "Join Break Lines"),
            ("lowercase", "Apply Lowercase"),
            ("strip_html", "Strip HTML tags"),
            ("remove_diacritics", "Diacritic Replacement"),
            ("normalize_spacing", "Normalize Spacing"),
            ("remove_stop_words", "Remove Stop Words"),
            ("word_tokenization", "Word Tokenization"),
            ("remove_greek", "Remove Greek letters"),
            ("remove_cyrillic", "Remove Cyrillic Script"),
            ("remove_super_sub_script", "Remove Superscript and Subscript Characters"),
            ("remove_bibliographical_references", "Remove Bibliographical References"),
            ("normalize_unicode", "Normalize Unicode"),
            ("remove_page_numbers", "Remove Page Numbers"),
        ]

        descriptions = {
            "remove_diacritics": "Replace diacritical marks (accents) from characters, standardizing the text (e.g., 'João' becomes 'Joao').",
            "normalize_spacing": "Adjust spacing to ensure consistent formatting throughout the text.",
            "remove_stop_words": "Eliminate common stop words (prepositions, determiners, conjunctions and non-meaningful words)\nto focus on significant words in the corpus.",
            "word_tokenization": "Split the text into individual words (tokens) for analysis.",
            "remove_bibliographical_references": "Eliminate bibliographical citations such as (Alves 1994) from the text.\nIt matches all occurrences that follow this pattern: '(' + capitalized word + any characters in between + a four-digit number + ')'.",
            "normalize_unicode": "Standardize text encoding to ensure consistency across the corpus.",
            "remove_page_numbers": "Delete standalone page numbers that are isolated on their own lines within the text.",
            "normalize_unicode": "Standardize text encoding by removing non-UTF-8 characters to ensure consistency across the corpus.",
            "remove_page_numbers": "Delete standalone page numbers that are isolated on their own lines within the text.",
        }

        tabs = QTabWidget()
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)

        vbox_layout = QVBoxLayout()
        for key, label in self.options:
            h_layout = QHBoxLayout()
            checkbox = QCheckBox(label)
            checkbox.setChecked(self.parameters.get(key, False))
            checkbox.stateChanged.connect(lambda state, k=key: self.parameters.update({k: bool(state)}))
            checkbox.setToolTip(f"Enable or disable {label.lower()}")
            h_layout.addWidget(checkbox)
            if key in descriptions:
                info_button = QToolButton()
                info_button.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_MessageBoxInformation')))
                info_button.setFixedSize(16, 16)
                info_button.setToolTip(descriptions[key])
                info_button.setStyleSheet("QToolButton { border: none; background-color: transparent; }")
                h_layout.addWidget(info_button)
            h_layout.addStretch()
            vbox_layout.addLayout(h_layout)
        
        general_layout.addLayout(vbox_layout)
        general_layout.addStretch()

        regex_button = QPushButton("Define Patterns with Regex")
        regex_button.clicked.connect(self.open_regex_dialog)
        regex_button.setToolTip("Define advanced regex patterns")
        advanced_layout.addWidget(regex_button)
        
        self.regex_display = QPlainTextEdit()
        self.regex_display.setReadOnly(True)
        self.regex_display.setPlainText(self.parameters.get("regex_pattern") or "None")
        self.regex_display.setFont(QFont("Courier New", 10))
        self.regex_display.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")
        self.regex_highlighter = RegexHighlighter(self.regex_display.document())
        
        advanced_layout.addWidget(QLabel("Current pattern:"))
        advanced_layout.addWidget(self.regex_display)
        
        char_remove_button = QPushButton("Select Characters or Sequences to Remove")
        char_remove_button.clicked.connect(self.open_char_selection)
        char_remove_button.setToolTip("Select specific characters or sequences to remove")
        
        advanced_layout.addWidget(char_remove_button)
        self.char_list_widget = QListWidget()
        self.update_char_list()
        
        advanced_layout.addWidget(QLabel("Selected items:"))
        advanced_layout.addWidget(self.char_list_widget)
        advanced_layout.addStretch()

        tabs.addTab(general_tab, "General")
        tabs.addTab(advanced_tab, "Advanced")
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.update_checkboxes()

    def open_regex_dialog(self):
        dialog = AdvancedPatternBuilder(self)
        if self.pattern_data:
            for row_data in self.pattern_data:
                dialog.addPattern()
                row_position = dialog.pattern_table.rowCount() - 1
                dialog.pattern_table.cellWidget(row_position, 0).setText(row_data["start"])
                dialog.pattern_table.cellWidget(row_position, 1).setCurrentText(row_data["end_type"])
                dialog.pattern_table.cellWidget(row_position, 2).setText(row_data["end"])
                dialog.pattern_table.cellWidget(row_position, 3).setText(row_data["number_length"])
        if dialog.exec():
            pattern = dialog.getPattern()
            if pattern:
                try:
                    re.compile(pattern.pattern)
                    self.parameters["regex_pattern"] = pattern.pattern
                    self.regex_display.setPlainText(pattern.pattern)
                    self.pattern_data = dialog.getPatternData()
                    self.parameters["pattern_data"] = self.pattern_data
                except re.error as e:
                    QMessageBox.warning(self, "Invalid Pattern", f"The entered pattern is invalid:\n{str(e)}")

    def open_char_selection(self):
        dialog = CharacterSelectionDialog(self.parameters.get("chars_to_remove", []), self)
        if dialog.exec():
            self.parameters["chars_to_remove"] = dialog.get_selected_chars()
            self.update_char_list()

    def update_char_list(self):
        self.char_list_widget.clear()
        for item in self.parameters.get("chars_to_remove", []):
            list_item = QListWidgetItem(item)
            self.char_list_widget.addItem(list_item)

    def update_checkboxes(self):
        for checkbox in self.findChildren(QCheckBox):
            key = next((k for k, v in self.options if v == checkbox.text()), None)
            if key:
                checkbox.setChecked(self.parameters.get(key, False))

    def get_parameters(self):
        return self.parameters
    
class CharacterSelectionDialog(QDialog):
    def __init__(self, current_chars, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Characeters or Sequences to Remove")
        self.setMinimumSize(400, 300)
        layout = QVBoxLayout(self)
        self.selected_chars = list(current_chars)
        input_layout = QHBoxLayout()
        self.char_input = QLineEdit()
        self.char_input.setPlaceholderText("Enter characters or sequences to remove")
        input_layout.addWidget(self.char_input)
        include_button = QPushButton("Include")
        include_button.clicked.connect(self.add_chars)
        include_button.setDefault(True)
        input_layout.addWidget(include_button)
        layout.addLayout(input_layout)
        self.char_list = QListWidget()
        self.update_char_list()
        layout.addWidget(QLabel("Items to remove:"))
        layout.addWidget(self.char_list)
        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self.delete_selected)
        layout.addWidget(delete_button)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        ok_button = button_box.button(QDialogButtonBox.Ok)
        if ok_button:
            ok_button.setAutoDefault(False)
            ok_button.setDefault(False)
        cancel_button = button_box.button(QDialogButtonBox.Cancel)
        if cancel_button:
            cancel_button.setAutoDefault(False)
            cancel_button.setDefault(False)
        self.char_input.returnPressed.connect(self.add_chars)

    def add_chars(self):
        new_item = self.char_input.text().strip()
        if new_item and new_item not in self.selected_chars:
            self.selected_chars.append(new_item)
            self.update_char_list()
        self.char_input.clear()

    def update_char_list(self):
        self.char_list.clear()
        for item in self.selected_chars:
            list_item = QListWidgetItem(item)
            self.char_list.addItem(list_item)

    def delete_selected(self):
        for item in self.char_list.selectedItems():
            self.selected_chars.remove(item.text())
        self.update_char_list()

    def get_selected_chars(self):
        return self.selected_chars

class FileLoadingDialog(QDialog):
    cancelled = Signal()

    def __init__(self, parent=None, title="Loading Files..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)
        self.label = QLabel("Loading files...")
        layout.addWidget(self.label)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)
        layout.addWidget(self.cancel_button)

    def update_progress(self, current, total, time_remaining, error):
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        if time_remaining is not None:
            self.label.setText(f"Processing file {current} of {total}... Estimated time remaining: {time_remaining:.2f} seconds")
        else:
            self.label.setText(f"Processing file {current} of {total}...")
        if error:
            pass

    def on_cancel(self):
        self.cancelled.emit()
        self.reject()

class DirectoryLoadingWorker(QObject):
    finished = Signal(list)

    def __init__(self, file_manager, directory, signals):
        super().__init__()
        self.file_manager = file_manager
        self.directory = directory
        self.signals = signals
        self._is_cancelled = False

    def run(self):
        new_files = self.file_manager.add_directory(self.directory, self.signals, self.is_cancelled)
        self.finished.emit(new_files)

    def cancel(self):
        self._is_cancelled = True

    def is_cancelled(self):
        return self._is_cancelled

class ReportWorker(QObject):
    progress = Signal(int)
    finished = Signal(dict)

    def __init__(self, files, parameters, processed=False, processed_results=None):
        super().__init__()
        self.files = files
        self.parameters = parameters
        self.processed = processed
        self.processed_results = processed_results
        self.nlp = get_spacy_model()
        self.batch_size = 100

    def run(self):
        try:
            total_words = 0
            total_size = 0
            file_count = len(self.files)

            for i in range(0, file_count, self.batch_size):
                batch = self.files[i:i+self.batch_size]
                batch_words, batch_size = self.process_batch(batch)
                
                total_words += batch_words
                total_size += batch_size

                progress = min(100, int((i + len(batch)) / file_count * 100))
                self.progress.emit(progress)

            avg_words = total_words / file_count if file_count else 0
            avg_size = total_size / file_count if file_count else 0

            final_report = {
                'total_files': file_count,
                'total_size': total_size / (1024 * 1024),
                'avg_size': avg_size / (1024 * 1024),
                'total_words': total_words,
                'avg_words': avg_words
            }

            self.finished.emit(final_report)
        except Exception as e:
            logging.error(f"Error in report generation: {str(e)}")

    def process_batch(self, batch):
        batch_words = 0
        batch_size = 0
        for file_path in batch:
            if self.processed and self.processed_results:
                _, _, text = next((r for r in self.processed_results if r[0] == file_path), (None, None, ''))
            else:
                # Detect encoding
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = from_bytes(raw_data)
                    best_guess = result.best()
                    if best_guess:
                        encoding = best_guess.encoding
                    else:
                        encoding = 'utf-8'  # Fallback

                with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                    text = file.read()

            batch_size += len(text.encode('utf-8'))

            if self.parameters.get("word_tokenization"):
                tokens = text.split()
                batch_words += len(tokens)
            else:
                doc = self.nlp(text)
                batch_words += len([token for token in doc if not token.is_space])

        return batch_words, batch_size
                                         
class WorkerSignals(QObject):
    result = Signal(str)
    error = Signal(str)
    finished = Signal()

class PreprocessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.version = "0.7"
        self.file_manager = FileManager()
        self.theme_manager = ThemeManager()
        self.processor = DocumentProcessor()
        self.signals = ProcessingSignals()
        self.current_file = None
        self.corpus_name = "Untitled Corpus"
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(multiprocessing.cpu_count())
        self.processed_results = []
        self.errors = []
        self.warnings = []
        self.items_per_page = 10  # Default value, can be overridden by user input
        self.files_processed = 0
        self.total_size = 0
        self.total_time = 0
        self.processed_size = 0
        self.documentation_window = None
        self.report_thread = QThread()
        self.report_worker = None
        self.lock = Lock()
        self.previous_search_term = ''
        self.init_ui()

    def init_ui(self):
        self.setFocusPolicy(Qt.StrongFocus)
        self.setWindowTitle('CorpuScript')
        self.setGeometry(100, 100, 1200, 800)
        self.setFont(QFont("Roboto", 10))
        icon_path = resource_path("my_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.create_menu_bar()
        self.create_toolbar()
        self.setup_central_widget()
        self.setup_status_bar()
        self.search_input.returnPressed.connect(self.search_text)
        self.prev_button.clicked.connect(self.go_to_previous_occurrence)
        self.next_button.clicked.connect(self.go_to_next_occurrence)
        self.signals.update_progress.connect(self.update_progress)
        self.signals.result.connect(self.handle_result)
        self.signals.error.connect(self.handle_error)
        self.signals.warning.connect(self.handle_warning)
        self.signals.finished.connect(self.on_worker_finished)
        self.signals.report_ready.connect(self.display_report)
        self.apply_theme()
        self.showMaximized()
        QTimer.singleShot(1000, lambda: self.check_for_updates(manual_trigger=False))

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        self.new_action = self.create_action("New", "document-new", "Ctrl+N", "Start a new project", self.confirm_start_new_cleaning)
        file_menu.addAction(self.new_action)
        self.open_files_action = self.create_action("Open Files", "document-open", "Ctrl+O", "Open files", self.open_file)
        file_menu.addAction(self.open_files_action)
        self.open_directory_action = self.create_action("Open Directory", "folder-open", "Ctrl+Shift+O", "Open directory", self.open_directory)
        file_menu.addAction(self.open_directory_action)
        self.save_action = self.create_action("Save", "document-save", "Ctrl+S", "Save current file", self.save_file)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        self.exit_action = self.create_action("Exit", "application-exit", "Ctrl+Q", "Exit the application", self.close)
        file_menu.addAction(self.exit_action)
        edit_menu = menu_bar.addMenu("&Edit")
        self.undo_action = self.create_action("Undo", "edit-undo", "Ctrl+Z", "Undo last action", self.undo)
        edit_menu.addAction(self.undo_action)
        self.redo_action = self.create_action("Redo", "edit-redo", "Ctrl+Y", "Redo last action", self.redo)
        edit_menu.addAction(self.redo_action)
        settings_menu = menu_bar.addMenu("&Settings")
        self.toggle_theme_action = self.create_action("Toggle Theme", "preferences-desktop-theme", "", "Switch between light and dark theme", self.toggle_theme)
        settings_menu.addAction(self.toggle_theme_action)
        self.processing_parameters_action = self.create_action("Processing Parameters", "preferences-system", "", "Configure processing options", self.open_parameters_dialog)
        settings_menu.addAction(self.processing_parameters_action)
        help_menu = menu_bar.addMenu("&Help")
        self.about_action = self.create_action("About", "help-about", "", "About this application", self.show_about_dialog)
        help_menu.addAction(self.about_action)
        self.documentation_action = self.create_action("Documentation", "help-contents", "F1", "View documentation", self.show_documentation)
        help_menu.addAction(self.documentation_action)
        self.check_updates_action = self.create_action("Check for Updates", "system-software-update", "", "Check for updates", lambda: self.check_for_updates(manual_trigger=True))
        help_menu.addAction(self.check_updates_action)
    
    def confirm_start_new_cleaning(self):
        reply = QMessageBox.question(self, 'Confirm New Project',
                                     "Are you sure you want to start a new project? All current data will be lost.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.start_new_cleaning()

    def start_new_cleaning(self):
        self.file_manager.clear_files()
        self.file_list.clear()

        self.original_text.clear()
        self.processed_text.clear()

        self.current_file = None
        self.corpus_name = "Untitled Corpus"

        self.processor.reset_parameters()

        self.clear_report()

        self.update_status_bar()
        self.update_report()

        if self.report_worker:
            self.report_thread.quit()
            self.report_thread.wait()
            self.report_worker = None

        QMessageBox.information(self, "New Project", "A new project has been started.")

    def clear_report(self):
        for key in self.report_items:
            self.report_items[key].findChild(QLabel, "reportValue").setText("0")
        self.summary_stack.setCurrentIndex(0)  

    def create_toolbar(self):
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)
        self.toolbar.addAction(self.new_action)
        self.toolbar.addAction(self.open_files_action)
        self.toolbar.addAction(self.open_directory_action)
        self.toolbar.addAction(self.save_action)
        self.toolbar.addSeparator()
        process_button = QPushButton("Process Files")
        process_button.setIcon(QIcon.fromTheme("system-run"))
        process_button.setToolTip("Process the selected files with current parameters")
        process_button.clicked.connect(self.process_files)
        self.toolbar.addWidget(process_button)

    def create_action(self, text, icon, shortcut, tooltip, callback):
        action = QAction(QIcon.fromTheme(icon, QIcon()), text, self)
        if shortcut:
            if sys.platform == "darwin":
                shortcut = shortcut.replace("Ctrl", "Meta")
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ApplicationShortcut)
        action.setToolTip(tooltip)
        action.triggered.connect(callback)
        return action
    
    def setup_central_widget(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left Panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(350)  # Set minimum width for visibility

        # File List Widget
        file_list_widget = QWidget()
        file_list_layout = QVBoxLayout(file_list_widget)
        selected_files_label = QLabel("Selected Files:")
        selected_files_label.setObjectName("sectionLabel")
        file_list_layout.addWidget(selected_files_label)

        self.file_list = FileListWidget()
        self.file_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.file_list.files_added.connect(lambda files: self.update_report())
        file_list_layout.addWidget(self.file_list)
        left_layout.addWidget(file_list_widget)

        # Report Widget
        report_widget = self.create_report_area()
        left_layout.addWidget(report_widget)

        left_layout.setStretch(0, 2)  # file_list_widget
        left_layout.setStretch(1, 1)  # report_widget

        # Text Display
        text_display = QWidget()
        text_layout = QVBoxLayout(text_display)
        text_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.text_tabs = QTabWidget()
        self.original_text = QPlainTextEdit()
        self.original_text.setReadOnly(True)
        self.processed_text = QPlainTextEdit()
        self.processed_text.setReadOnly(True)
        self.text_tabs.addTab(self.original_text, "Original Text")
        self.text_tabs.addTab(self.processed_text, "Processed Text")
        text_layout.addWidget(self.text_tabs)

        # Search Layout
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search in text...")
        self.search_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        search_layout.addWidget(self.search_input)

        self.prev_button = QPushButton()
        self.prev_button.setIcon(QIcon.fromTheme("go-previous"))
        self.prev_button.setToolTip("Previous occurrence")
        self.prev_button.clicked.connect(self.go_to_previous_occurrence)
        self.prev_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        search_layout.addWidget(self.prev_button)

        self.next_button = QPushButton()
        self.next_button.setIcon(QIcon.fromTheme("go-next"))
        self.next_button.setToolTip("Next occurrence")
        self.next_button.clicked.connect(self.go_to_next_occurrence)
        self.next_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        search_layout.addWidget(self.next_button)

        self.occurrence_label = QLabel("0/0")
        self.occurrence_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        search_layout.addWidget(self.occurrence_label)

        text_layout.addLayout(search_layout)

        self.current_occurrence_index = -1
        self.search_results = []

        # Splitter to allow resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(text_display)
        splitter.setStretchFactor(0, 1)  # Left panel 
        splitter.setStretchFactor(1, 2)  # Right panel (text display) gets more space

        main_layout.addWidget(splitter)

        self.setCentralWidget(central_widget)

    def create_report_area(self):
        outer_widget = QWidget()
        outer_layout = QVBoxLayout(outer_widget)

        report_label = QLabel("Summary Report:")
        report_label.setObjectName("sectionLabel")
        outer_layout.addWidget(report_label)

        report_widget = QWidget()
        report_widget.setObjectName("reportWidget")
        report_layout = QVBoxLayout(report_widget)

        self.summary_stack = QStackedWidget()
        
        empty_widget = QWidget()
        self.summary_stack.addWidget(empty_widget)

        loading_widget = QWidget()
        loading_layout = QVBoxLayout(loading_widget)
        self.loading_label = QLabel("Generating report...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        loading_layout.addWidget(self.loading_label)
        self.report_progress_bar = QProgressBar()
        self.report_progress_bar.setRange(0, 100)
        self.report_progress_bar.setValue(0)
        loading_layout.addWidget(self.report_progress_bar)
        self.summary_stack.addWidget(loading_widget)

        self.report_grid = QGridLayout()
        self.report_grid.setSpacing(10)

        self.report_items = {
            'total_files': self.create_report_item("Total Files Processed", "0"),
            'total_size': self.create_report_item("Total Size of Processed Files", "0 MB"),
            'avg_size': self.create_report_item("Average File Size", "0 MB"),
            'total_words': self.create_report_item("Total Word Count", "0"),
            'avg_words': self.create_report_item("Average Word Count per File", "0")
        }

        positions = [(i, j) for i in range(3) for j in range(2)]
        for (key, widget), position in zip(self.report_items.items(), positions):
            self.report_grid.addWidget(widget, *position)

        report_content = QWidget()
        report_content.setLayout(self.report_grid)
        self.summary_stack.addWidget(report_content)

        report_layout.addWidget(self.summary_stack)
        
        outer_layout.addWidget(report_widget)

        return outer_widget

    def create_report_item(self, label, value):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        value_label = QLabel(value)
        value_label.setObjectName("reportValue")
        value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        description_label = QLabel(label)
        description_label.setObjectName("reportDescription")
        description_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        layout.addWidget(value_label)
        layout.addWidget(description_label)
        
        return widget         
    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status_bar()

    def get_report_stylesheet(self):
        return """
            QWidget#reportWidget {
                background-color: palette(base);
                border-radius: 2px;
            }
            QLabel#reportValue {
                font-size: 24px;
                font-weight: bold;
                color: palette(text);
            }
            QLabel#reportDescription {
                font-size: 12px;
                color: palette(mid);
            }
        """
    def apply_theme(self):
        self.setStyleSheet(self.theme_manager.get_stylesheet())
        self.update_icon_colors()

    def update_icon_colors(self):
        icon_color = QColor(self.theme_manager.custom_colors['icon_dark' if self.theme_manager.dark_theme else 'icon_light'])
        for action in self.toolbar.actions():
            if not action.isSeparator():
                icon = action.icon()
                if not icon.isNull():
                    pixmap = icon.pixmap(QSize(24, 24))
                    if not pixmap.isNull():
                        painter = QPainter(pixmap)
                        if painter.isActive():
                            painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                            painter.fillRect(pixmap.rect(), icon_color)
                            painter.end()
                            action.setIcon(QIcon(pixmap))

    def toggle_theme(self):
        self.theme_manager.toggle_theme()
        self.apply_theme()

    def update_status_bar(self):
        total_size_mb = sum(os.path.getsize(file) for file in self.file_manager.get_files()) / (1024 * 1024)
        status_text = f"Files: {len(self.file_manager.get_files())} | Total Size: {total_size_mb:.2f} MB | Status: {'Processing' if self.thread_pool.activeThreadCount() > 0 else 'Idle'}"
        self.status_bar.showMessage(status_text)
        if 'Idle' in status_text:
            self.status_bar.clearMessage()

    def open_file(self):
        self.status_bar.clearMessage()
        files, _ = QFileDialog.getOpenFileNames(self, "Select files", "", "Text Files (*.txt)")
        if files:
            new_files = self.file_manager.add_files(files)
            if new_files:
                self.file_list.addItems(new_files)
                self.update_status_bar()
                self.process_files()
            else:
                QMessageBox.information(self, "No New Files", "All selected files are already loaded.")

    def open_directory(self):
        self.status_bar.clearMessage()
        if hasattr(self, 'loading_thread') and self.loading_thread.isRunning():
            self.cancel_loading()
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.loading_dialog = FileLoadingDialog(self)
            self.loading_dialog.show()
            self.signals.update_progress.connect(self.loading_dialog.update_progress)
            self.loading_thread = QThread()
            self.loading_worker = DirectoryLoadingWorker(self.file_manager, directory, self.signals)
            self.loading_worker.moveToThread(self.loading_thread)
            self.loading_thread.started.connect(self.loading_worker.run)
            self.loading_worker.finished.connect(self.on_directory_loading_finished)
            self.loading_dialog.cancelled.connect(self.cancel_loading)
            self.loading_dialog.rejected.connect(self.cancel_loading)

            self.loading_thread.start()

    def cancel_loading(self):
        if hasattr(self, 'loading_worker'):
            self.loading_worker.cancel()
        if hasattr(self, 'loading_thread'):
            self.loading_thread.quit()
            self.loading_thread.wait()
        self.loading_dialog.close()

    def on_directory_loading_finished(self, new_files):
        self.loading_thread.quit()
        self.loading_thread.wait()
        self.loading_dialog.close()
        if new_files:
            self.file_list.addItems(new_files)
            self.update_status_bar()
            self.process_files()
        else:
            self.status_bar.showMessage("Operation cancelled or no new files added.", 5000)
    
    def save_file(self):
        if self.processed_results:
            reply = QMessageBox.question(
                self, 'Confirm Save',
                "This will overwrite the original files. Do you want to proceed?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                failed_files = []
                for file_path, _, processed_text in self.processed_results:
                    try:
                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.write(processed_text)
                    except Exception as e:
                        logging.error(f"Error saving file {file_path}: {str(e)}")
                        failed_files.append(file_path)
                if failed_files:
                    QMessageBox.warning(self, 'Save Completed with Errors',
                                        f"Some files could not be saved:\n{', '.join(failed_files)}")
                else:
                    QMessageBox.information(self, 'Save Successful', "All files have been saved successfully.")
        else:
            QMessageBox.warning(self, 'Save Failed', "No processed files to save.")
            
    def process_files(self):
        self.status_bar.clearMessage()
        if not self.file_manager.get_files():
            QMessageBox.warning(self, "No Files", "Please select files to process.")
            return

        total_files = len(self.file_manager.get_files())
        dialog = FileDisplayDialog(total_files, self)
        if dialog.exec():
            number_to_display = dialog.get_number_of_files()
            self.items_per_page = number_to_display
        else:
            # User cancelled the dialog
            return

        # Proceed with processing
        self.processing_dialog = FileLoadingDialog(self, title="Processing Files...")
        self.processing_dialog.show()
        self.signals.update_progress.connect(self.processing_dialog.update_progress)
        self.processed_results.clear()
        self.errors.clear()
        self.warnings.clear()
        self.files_processed = 0
        self.total_size = self.file_manager.get_total_size()
        self.total_time = 0
        self.processed_size = 0

        for file in self.file_manager.get_files():
            worker = ProcessingWorker(self.processor, file, self.signals)
            self.thread_pool.start(worker)

    def update_progress_info(self, message=None, error=None):
        progress = (self.files_processed / len(self.file_manager.get_files())) * 100 if self.file_manager.get_files() else 0
        avg_speed = self.processed_size / self.total_time if self.total_time > 0 else 0
        status_message = f"Progress: {progress:.2f}% | Avg. Speed: {avg_speed:.2f} B/s"
        if message:
            status_message += f" | {message}"
        if error:
            status_message += f" | Error: {error}"
        self.status_bar.showMessage(status_message)

    def handle_result(self, file_path, original_text, processed_text, file_size, processing_time):  
        with self.lock:
            self.processed_results.append((file_path, original_text, processed_text))  
            self.total_time += processing_time
            self.files_processed += 1
            self.processed_size += file_size
        remaining_files = len(self.file_manager.get_files()) - self.files_processed
        self.update_progress_info(message=f"Processed {os.path.basename(file_path)} | Remaining files: {remaining_files}")

    def handle_error(self, file_path, error):
        with self.lock:
            self.errors.append((file_path, error))
            self.files_processed += 1
        remaining_files = len(self.file_manager.get_files()) - self.files_processed
        error_message = f"Error in {os.path.basename(file_path)}: {error} | Remaining files: {remaining_files}"
        self.update_progress_info(error=error_message)
        logging.error(f"Error processing {file_path}: {error}")

    def handle_warning(self, file_path, warning):
        with self.lock:
            self.warnings.append((file_path, warning))
        warning_message = f"Warning in {os.path.basename(file_path)}: {warning}"
        self.update_progress_info(message=warning_message)
        logging.warning(f"Warning processing {file_path}: {warning}")

    def update_progress(self, current, total, time_remaining, error):
        with self.lock:
            self.files_processed = current
        self.update_progress_info()
        if error:
            self.status_bar.showMessage(f"Error: {error}", 5000)
            logging.error(f"Error during loading: {error}")

    def on_worker_finished(self):
        if self.files_processed == len(self.file_manager.get_files()):
            self.signals.processing_complete.emit(self.processed_results, self.warnings)
            if self.errors:
                error_msg = "\n".join([f"{file}: {error}" for file, error in self.errors])
                QMessageBox.warning(self, 'Processing Completed with Errors', f"Errors occurred during processing:\n\n{error_msg}")
            else:
                self.status_bar.clearMessage()
            self.update_status_bar()
            self.processing_dialog.close()
            self.display_results(self.processed_results, self.warnings)
            self._generate_report(processed=True, processed_results=self.processed_results)

    def display_results(self, results, warnings):
        self.results = results  # Store results
        self.current_page = 0
        self.display_page(items_per_page=self.items_per_page)  # Display specified number of files

        if warnings:
            warning_msg = "\n".join([f"{file}: {warning}" for file, warning in warnings])
            QMessageBox.warning(self, 'Processing Warnings', f"Warnings during processing:\n\n{warning_msg}")

    def display_page(self, page_number=0, items_per_page=None):
        if items_per_page is None:
            page_results = self.results  # Display all results
        else:
            start = page_number * items_per_page
            end = start + items_per_page
            page_results = self.results[start:end]

        self.original_text.clear()
        self.processed_text.clear()

        for file_path, original_text, processed_text in page_results:
            self.original_text.appendPlainText(f"File: {file_path}\n\n{original_text}\n\n")
            self.processed_text.appendPlainText(f"File: {file_path}\n\n{processed_text}\n\n")
            
    def display_original_texts(self):
        if self.file_manager.get_files():
            self.original_text.clear()
            self.processed_text.clear()
            for file_path in self.file_manager.get_files():
                try:
                    # Encoding detection
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                        result = from_bytes(raw_data)
                        best_guess = result.best()
                        if best_guess:
                            encoding = best_guess.encoding
                        else:
                            encoding = 'utf-8'  # Fallback encoding

                    # Read with detected encoding
                    with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                        original_text = file.read()

                    self.original_text.appendPlainText(f"File: {file_path}\n\n{original_text}\n\n")
                    self.processed_text.appendPlainText(f"File: {file_path}\n\n{original_text}\n\n")
                except Exception as e:
                    logging.error(f"Error reading file {file_path}: {str(e)}")
            self.current_file = self.file_manager.get_files()[0]
        else:
            self.original_text.clear()
            self.processed_text.clear()
            self.current_file = None

    def start_new_cleaning(self):
        self.file_manager.clear_files()
        self.file_list.clear()
        self.original_text.clear()
        self.processed_text.clear()
        self.current_file = None
        self.corpus_name = "Untitled Corpus"
        self.update_status_bar()

        if self.report_worker:
            self.report_thread.quit()
            self.report_thread.wait()
            self.report_worker = None

        self.update_report()  

    def open_parameters_dialog(self):
        dialog = ParametersDialog(self.processor.get_parameters(), self)
        if dialog.exec():
            new_parameters = dialog.get_parameters()
            self.processor.set_parameters(new_parameters)
            self.processor.update_pipeline()

    def show_about_dialog(self):
        QMessageBox.about(self, "About", f"CorpuScript\nVersion {self.version}\n\nDeveloped by Jhonatan Lopes")

    def show_documentation(self):
        documentation_file = resource_path("documentation.html")
        if not os.path.exists(documentation_file):
            QMessageBox.warning(self, "Documentation Not Found", f"The documentation.html file could not be found: {documentation_file}")
            return
        with open(documentation_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        base_url = QUrl.fromLocalFile(os.path.abspath(documentation_file))

        bg_color = self.theme_manager.custom_colors['background']
        text_color = self.theme_manager.custom_colors['text']
        primary_color = self.theme_manager.custom_colors['primary']
        secondary_color = self.theme_manager.custom_colors['secondary']
        accent_color = self.theme_manager.custom_colors['accent']

        css_styles = f"""
        <style>
            body {{
                background-color: {bg_color};
                color: {text_color};
                font-family: Roboto, sans-serif;
                margin: 20px;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: {primary_color};
            }}
            a {{
                color: {secondary_color};
            }}
            code, pre {{
                background-color: #444444;
                color: #f0f0f0;
                padding: 5px;
                border-radius: 5px;
                font-family: 'Courier New', Courier, monospace;
            }}
            pre {{
                padding: 10px;
            }}
        </style>
        """

        html_content = html_content.replace('</head>', f'{css_styles}</head>')

        if self.documentation_window is None:
            self.documentation_window = QWebEngineView()
            self.documentation_window.setWindowTitle("Documentation")
            self.documentation_window.resize(800, 600)
            self.documentation_window.setMinimumSize(800, 600)
            self.documentation_window.setWindowIcon(self.windowIcon())

        self.documentation_window.setStyleSheet(self.theme_manager.get_stylesheet())
        self.documentation_window.setHtml(html_content, base_url)
        self.documentation_window.show()

    def goto_occurrence(self, text_edit, index):
        if 0 <= index < len(self.search_results):
            cursor = QTextCursor(text_edit.document())
            cursor.setPosition(self.search_results[index].selectionStart())
            text_edit.setTextCursor(cursor)
            text_edit.centerCursor()

    def highlight_search_term(self, text_edit, search_term, current_index=-1):

        extra_selections = []
        self.search_results = []

        if not search_term:
            text_edit.setExtraSelections(extra_selections)
            self.update_occurrence_label()
            return

        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QColor("#FFFF00"))  # Yellow background
        highlight_format.setForeground(QColor("#000000"))  # Black text
        highlight_format.setFontWeight(QFont.Bold)

        current_highlight_format = QTextCharFormat()
        current_highlight_format.setBackground(QColor("#FFA500"))  # Orange background
        current_highlight_format.setForeground(QColor("#000000"))  # Black text
        current_highlight_format.setFontWeight(QFont.Bold)

        document = text_edit.document()
        cursor = QTextCursor(document)

        regex_pattern = QRegularExpression.escape(search_term)
        regex = QRegularExpression(regex_pattern)
        regex.setPatternOptions(QRegularExpression.CaseInsensitiveOption)  

        matches = regex.globalMatch(document.toPlainText())

        while matches.hasNext():
            match = matches.next()
            match_cursor = QTextCursor(document)
            match_cursor.setPosition(match.capturedStart())
            match_cursor.setPosition(match.capturedEnd(), QTextCursor.KeepAnchor)

            selection = QTextEdit.ExtraSelection()
            selection.cursor = match_cursor

            if len(self.search_results) == current_index:
                selection.format = current_highlight_format
            else:
                selection.format = highlight_format

            extra_selections.append(selection)
            self.search_results.append(match_cursor)

        text_edit.setExtraSelections(extra_selections)
        self.update_occurrence_label()

    def search_text(self):
        search_term = self.search_input.text()
        active_text_edit = self.text_tabs.currentWidget()
        if not isinstance(active_text_edit, QPlainTextEdit):
            return

        if search_term != self.previous_search_term:
            self.previous_search_term = search_term
            self.current_occurrence_index = 0
        else:
            self.go_to_next_occurrence()
            return

        self.highlight_search_term(active_text_edit, search_term, self.current_occurrence_index)
        if self.search_results:
            self.goto_occurrence(active_text_edit, self.current_occurrence_index)
            self.update_occurrence_label()
        else:
            self.current_occurrence_index = -1
            self.update_occurrence_label()
            QMessageBox.information(self, "Not Found", f"The text '{search_term}' was not found.")

    def go_to_next_occurrence(self):
        search_term = self.search_input.text()
        active_text_edit = self.text_tabs.currentWidget()
        if not isinstance(active_text_edit, QPlainTextEdit):
            return
        if not self.search_results or search_term != self.previous_search_term:
            self.search_text()
        else:
            if self.search_results:
                self.current_occurrence_index = (self.current_occurrence_index + 1) % len(self.search_results)
                self.highlight_search_term(active_text_edit, search_term, self.current_occurrence_index)
                self.goto_occurrence(active_text_edit, self.current_occurrence_index)
                self.update_occurrence_label()

    def go_to_previous_occurrence(self):
        search_term = self.search_input.text()
        active_text_edit = self.text_tabs.currentWidget()
        if not isinstance(active_text_edit, QPlainTextEdit):
            return
        if not self.search_results or search_term != self.previous_search_term:
            self.search_text()
        else:
            if self.search_results:
                self.current_occurrence_index = (self.current_occurrence_index - 1) % len(self.search_results)
                self.highlight_search_term(active_text_edit, search_term, self.current_occurrence_index)
                self.goto_occurrence(active_text_edit, self.current_occurrence_index)
                self.update_occurrence_label()

    def update_occurrence_label(self):
        total = len(self.search_results)
        current = self.current_occurrence_index + 1 if self.current_occurrence_index >= 0 else 0
        self.occurrence_label.setText(f"{current}/{total}")

    def undo(self):
        active_text_edit = self.text_tabs.currentWidget()
        if isinstance(active_text_edit, QPlainTextEdit):
            active_text_edit.undo()

    def redo(self):
        active_text_edit = self.text_tabs.currentWidget()
        if isinstance(active_text_edit, QPlainTextEdit):
            active_text_edit.redo()

    def update_report_display(self, report_data):
        self.summary_stack.setCurrentIndex(2)  # Switch to the report content
        for key, value in report_data.items():
            if key in self.report_items:
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                if key.endswith('_size'):
                    formatted_value += " MB"
                
                self.report_items[key].findChild(QLabel, "reportValue").setText(formatted_value)
                
    def _generate_report(self, processed=False, processed_results=None):
        if self.report_worker is not None:
            return
        
        parameters = self.processor.get_parameters()
        self.report_worker = ReportWorker(self.file_manager.get_files(), parameters, processed, processed_results)
        self.report_thread = QThread()
        self.report_worker.moveToThread(self.report_thread)
        self.report_worker.progress.connect(self.update_report_progress)
        self.report_worker.finished.connect(self.display_final_report)
        
        self.report_thread.started.connect(self.report_worker.run)
        self.report_thread.start()

        # Show the loading widget while the report is being generated
        self.summary_stack.setCurrentIndex(1)
                    
    def display_final_report(self, report_data):
        self.summary_stack.setCurrentIndex(2)  # Switch to the report content
        for key, value in report_data.items():
            if key in self.report_items:
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                if key.endswith('_size'):
                    formatted_value += " MB"
                
                self.report_items[key].findChild(QLabel, "reportValue").setText(formatted_value)
        
        self._cleanup_report_worker()

    def _cleanup_report_worker(self):
        if self.report_worker:
            self.report_worker.deleteLater()
            self.report_worker = None
        if self.report_thread:
            self.report_thread.quit()
            self.report_thread.wait()
            self.report_thread.deleteLater()
            self.report_thread = None

    def update_report_progress(self, progress):
        self.report_progress_bar.setValue(progress)
        self.loading_label.setText(f"Generating report... {progress}%")

    def handle_report_error(self, error_msg):
        logging.error(f"Error in report generation: {error_msg}")
        self.status_bar.showMessage("Error generating report", 5000)
        # Switch back to summary_text and display the error message
        self.summary_stack.setCurrentWidget(self.summary_text)
        if hasattr(self, 'loading_movie'):
            self.loading_movie.stop()
        self.summary_text.setHtml(f"<div style='color: #FF0000;'>Error generating report: {error_msg}</div>")

    def on_report_finished(self):
        logging.info("Report generation finished")
        if hasattr(self, 'report_progress_bar'):
            self.report_progress_bar.setValue(100)
        self.loading_label.setText("Report generation complete")
        self.report_thread.quit()
        self.report_thread.wait()
        self._cleanup_report_worker()

    def display_report(self, report_data):
        self.summary_stack.setCurrentIndex(2)  # Switch to the report content
        self.report_items['total_files'].findChild(QLabel, "reportValue").setText(str(report_data['total_files']))
        self.report_items['total_size'].findChild(QLabel, "reportValue").setText(f"{report_data['total_size']:.2f} MB")
        self.report_items['avg_size'].findChild(QLabel, "reportValue").setText(f"{report_data['avg_size']:.2f} MB")
        self.report_items['total_words'].findChild(QLabel, "reportValue").setText(str(report_data['total_words']))
        self.report_items['avg_words'].findChild(QLabel, "reportValue").setText(f"{report_data['avg_words']:.2f}")
    
    def check_for_updates(self, manual_trigger=False):
        try:
            url = 'https://api.github.com/repos/jhlopesalves/CorpuScript/releases/latest'
            with urllib.request.urlopen(url) as response:
                data = response.read()
                latest_release = json.loads(data.decode('utf-8'))
                latest_version = latest_release['tag_name']

                def parse_version(version_str):
                    return tuple(map(int, (version_str.strip('v').split('.'))))
                
                current_version = parse_version(self.version)
                latest_version_parsed = parse_version(latest_version)

                if latest_version_parsed > current_version:
                    reply = QMessageBox.question(
                        self, 'Update Available',
                        f"A new version {latest_version} is available. Do you want to download it?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                    )
                    if reply == QMessageBox.Yes:
                        QDesktopServices.openUrl(QUrl(latest_release['html_url']))
                elif manual_trigger:
                    QMessageBox.information(self, 'Up-to-Date', "You are using the latest version.")
        except urllib.error.HTTPError as e:
            if e.code == 404 and manual_trigger:
                QMessageBox.information(self, 'No Releases', "No updates are available yet. You are using the latest version.")
            elif e.code != 404:
                QMessageBox.warning(self, 'Error', f"An error occurred while checking for updates:\n{str(e)}")
        except Exception as e:
            QMessageBox.warning(self, 'Error', f"An error occurred while checking for updates:\n{str(e)}")

    def closeEvent(self, event):
        self.thread_pool.clear()
        self.thread_pool.waitForDone()
        if hasattr(self, 'loading_thread') and self.loading_thread.isRunning():
            self.cancel_loading()
        if self.report_thread:
            self.report_thread.quit()
            self.report_thread.wait()
        event.accept()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "CorpuScript.log")
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Application started.")
    try:
        nlp = get_spacy_model()
        logging.info("spaCy model loaded successfully.")
        
        app = QApplication(sys.argv)
        icon_path = resource_path("my_icon.ico")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
        window = PreprocessorGUI()
        window.show()
        exit_code = app.exec()
        window.cleanup()  
        logging.info("Main window shown.")
        sys.exit(app.exec())
    except Exception as e:
        logging.exception("An error occurred: %s", e)

if __name__ == "__main__":
    main()