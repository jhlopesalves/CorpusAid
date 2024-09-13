import os
import sys
import re
import unicodedata
import logging
import time
import random
import multiprocessing
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                               QTextEdit, QProgressBar, QFileDialog, QLabel, QListWidget,
                               QTabWidget, QLineEdit, QDialog, QDialogButtonBox, QCheckBox, QMessageBox,
                               QScrollArea, QSplitter, QToolBar, QStatusBar, QListWidgetItem,
                               QPlainTextEdit, QWizard, QWizardPage, QTableWidget, QComboBox)
from PySide6.QtCore import Qt, Signal, QObject, QThread, QSize, QRunnable, QThreadPool
from PySide6.QtGui import QIcon, QFont, QColor, QAction, QPainter, QIntValidator, QTextOption, QTextCursor, QTextCharFormat
from bs4 import BeautifulSoup
from collections import Counter

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_version():
    try:
        with open(resource_path("version.txt"), "r") as f:
            return f.read().strip()
    except:
        return "0.2"

def ensure_nltk_data():
    required_data = [
        'tokenizers/punkt',
        'corpora/stopwords',
    ]
    for item in required_data:
        try:
            import nltk
            nltk.data.find(item)
        except (LookupError, ImportError):
            import nltk
            nltk.download(item.split('/')[-1])

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

class WhitespaceNormalizationModule(PreprocessingModule):
    def process(self, text):
        text = re.sub(r'\s+([.,?!;:])', r'\1', text)
        text = re.sub(r'\s*([.,?!;:])\s*', r'\1 ', text)
        text = re.sub(r'\s*([\[\]{}\(\)/])\s*', r' \1 ', text)
        text = re.sub(r'(\d)\s*([%$£€])', r'\1\2', text)
        text = re.sub(r'(\d)\s*([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'\s*([+\-*/=])\s*', r' \1 ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        text = re.sub(r'([a-zA-Z])\s+([%$£€])', r'\1\2', text)
        text = re.sub(r'([%$£€])\s+(\d)', r'\1\2', text)
        text = re.sub(r'(\d)\s+([.,?!;:])', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

class LineBreakRemovalModule(PreprocessingModule):
    def process(self, text):
        return re.sub(r'\n', ' ', text)

class LowercaseModule(PreprocessingModule):
    def process(self, text):
        return text.lower()

class StopWordRemovalModule(PreprocessingModule):
    def __init__(self):
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))
    def process(self, text):
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text
        result = [word for word in tokens if word.lower() not in self.stop_words]
        return result

class RegexFilterModule(PreprocessingModule):
    def __init__(self, pattern, replacement=''):
        self.pattern = re.compile(pattern) if pattern else None
        self.replacement = replacement
    def process(self, text):
        if self.pattern:
            result = self.pattern.sub(self.replacement, text)
            return result
        return text

class HTMLStripperModule(PreprocessingModule):
    def process(self, text):
        return BeautifulSoup(text, "html.parser").get_text()

class DiacriticRemovalModule(PreprocessingModule):
    def process(self, text):
        return ''.join(c for c in unicodedata.normalize('NFD', text)
                       if unicodedata.category(c) != 'Mn')

class GreekLetterRemovalModule(PreprocessingModule):
    def process(self, text):
        return ''.join(char for char in text if not unicodedata.name(char, '').startswith('GREEK'))

class CyrillicRemovalModule(PreprocessingModule):
    def process(self, text):
        return ''.join(char for char in text if not unicodedata.name(char, '').startswith('CYRILLIC'))

class SuperSubScriptRemovalModule(PreprocessingModule):
    def process(self, text):
        return ''.join([char for char in text if unicodedata.category(char) not in ['Ps', 'Pi', 'Pf', 'Pd']])

class PreprocessingPipeline:
    def __init__(self):
        self.modules = []
    def add_module(self, module):
        self.modules.append(module)
    def process(self, text):
        requires_tokenization = any(isinstance(module, StopWordRemovalModule) for module in self.modules)
        if requires_tokenization:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text)
            for module in self.modules:
                if isinstance(module, StopWordRemovalModule):
                    tokens = module.process(tokens)
                else:
                    text = module.process(text)
                    tokens = word_tokenize(text)
            result = ' '.join(tokens)
        else:
            for module in self.modules:
                text = module.process(text)
            result = text
        return result.strip()

class DocumentProcessor:
    def __init__(self):
        self.pipeline = PreprocessingPipeline()
        self.parameters = {
            "remove_break_lines": False,
            "lowercase": False,
            "chars_to_remove": [],
            "remove_stop_words": False,
            "regex_pattern": "",
            "strip_html": False,
            "remove_diacritics": False,
            "remove_greek": False,
            "remove_cyrillic": False,
            "remove_super_sub_script": False,
            "normalize_spacing": False,
            "pattern_data": []
        }
    def set_parameters(self, parameters):
        try:
            if "regex_pattern" in parameters:
                re.compile(parameters["regex_pattern"])
            self.parameters.update(parameters)
            self.update_pipeline()
        except re.error:
            pass
    def update_pipeline(self):
        self.pipeline = PreprocessingPipeline()
        if self.parameters["regex_pattern"]:
            self.pipeline.add_module(RegexFilterModule(self.parameters["regex_pattern"]))
        if self.parameters["strip_html"]:
            self.pipeline.add_module(HTMLStripperModule())
        if self.parameters["remove_break_lines"]:
            self.pipeline.add_module(LineBreakRemovalModule())
        if self.parameters["lowercase"]:
            self.pipeline.add_module(LowercaseModule())
        if self.parameters["chars_to_remove"]:
            self.pipeline.add_module(CharacterFilterModule(self.parameters["chars_to_remove"]))
        if self.parameters["remove_stop_words"]:
            self.pipeline.add_module(StopWordRemovalModule())
        if self.parameters["remove_diacritics"]:
            self.pipeline.add_module(DiacriticRemovalModule())
        if self.parameters["remove_greek"]:
            self.pipeline.add_module(GreekLetterRemovalModule())
        if self.parameters["remove_cyrillic"]:
            self.pipeline.add_module(CyrillicRemovalModule())
        if self.parameters["remove_super_sub_script"]:
            self.pipeline.add_module(SuperSubScriptRemovalModule())
        if self.parameters["normalize_spacing"]:
            self.pipeline.add_module(WhitespaceNormalizationModule())
    def get_parameters(self):
        return self.parameters
    def process_file(self, text):
        for module in self.pipeline.modules:
            text = module.process(text)
            if isinstance(text, list):
                text = ' '.join(text)
        return text

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
            start_time = time.time()
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
            processed_text = self.processor.process_file(text)
            end_time = time.time()
            processing_time = end_time - start_time
            self.signals.result.emit(self.file_path, processed_text, file_size, processing_time)
        except Exception as e:
            self.signals.error.emit(self.file_path, str(e))
        finally:
            self.signals.finished.emit()

class ProcessingSignals(QObject):
    result = Signal(str, str, int, float)
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
    def add_directory(self, directory, signals):
        new_files = []
        total_files = 0
        for root, dirs, files in os.walk(directory):
            total_files += len([f for f in files if f.endswith(".txt")])
        start_time = time.time()
        processed_files = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    normalized_path = os.path.normpath(file_path)
                    if normalized_path not in self.files:
                        try:
                            time.sleep(random.uniform(0.01, 0.1))
                            self.files.append(normalized_path)
                            new_files.append(normalized_path)
                            processed_files += 1
                            elapsed_time = time.time() - start_time
                            estimated_remaining_time = (elapsed_time / processed_files) * (total_files - processed_files) if processed_files > 0 else 0
                            signals.update_progress.emit(processed_files, total_files, estimated_remaining_time, None)
                        except Exception as e:
                            signals.update_progress.emit(processed_files, total_files, 0, str(e))
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
            "icon_light": "#000000"
        }
    def toggle_theme(self):
        self.dark_theme = not self.dark_theme
    def get_stylesheet(self):
        if self.dark_theme:
            return self._get_dark_stylesheet()
        else:
            return self._get_light_stylesheet()
    def _get_dark_stylesheet(self):
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
            QPushButton:pressed {{
                background-color: {self.custom_colors['accent']};
            }}
            QLineEdit, QTextEdit, QListWidget, QPlainTextEdit {{
                background-color: #2D2D2D;
                border: 1px solid #3F3F3F;
                padding: 3px;
            }}
            QToolBar {{
                border: 1px solid #3F3F3F;
            }}
            QMenu {{
                background-color: #2D2D2D;
                border: 1px solid #3F3F3F;
            }}
            QMenu::item:selected {{
                background-color: {self.custom_colors['primary']};
            }}
            QTabWidget::pane {{
                border-top: 2px solid #3F3F3F;
            }}
            QTabBar::tab {{
                background: #2D2D2D;
                color: {self.custom_colors['text']};
                padding: 5px 10px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }}
            QTabBar::tab:selected {{
                background: {self.custom_colors['primary']};
            }}
        """
    def _get_light_stylesheet(self):
        return f"""
            QMainWindow, QWidget {{
                background-color: #F0F0F0;
                color: #000000;
            }}
            QPushButton {{
                background-color: {self.custom_colors['primary']};
                color: #FFFFFF;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {self.custom_colors['secondary']};
            }}
            QPushButton:pressed {{
                background-color: {self.custom_colors['accent']};
            }}
            QLineEdit, QTextEdit, QListWidget, QPlainTextEdit {{
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                padding: 3px;
            }}
            QToolBar {{
                border: 1px solid #CCCCCC;
            }}
            QMenu {{
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
            }}
            QMenu::item:selected {{
                background-color: {self.custom_colors['primary']};
                color: #FFFFFF;
            }}
            QTabWidget::pane {{
                border-top: 2px solid #CCCCCC;
            }}
            QTabBar::tab {{
                background: #E1E1E1;
                color: #000000;
                padding: 5px 10px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }}
            QTabBar::tab:selected {{
                background: {self.custom_colors['primary']};
                color: #FFFFFF;
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
        layout.addWidget(self.pattern_table)
        add_button = QPushButton("Add Pattern")
        add_button.clicked.connect(self.addPattern)
        layout.addWidget(add_button)
        options_layout = QHBoxLayout()
        self.case_sensitive = QCheckBox("Case sensitive")
        options_layout.addWidget(self.case_sensitive)
        self.whole_words = QCheckBox("Match whole words only")
        options_layout.addWidget(self.whole_words)
        layout.addLayout(options_layout)
        page.setLayout(layout)
        return page

    def createPreviewPage(self):
        page = QWizardPage()
        page.setTitle("Preview and Test")
        layout = QVBoxLayout()
        self.pattern_preview = QLineEdit()
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
        end_type_combo.addItems(["Single Number", "Multiple Numbers", "Specific Word"])
        self.pattern_table.setCellWidget(row_position, 1, end_type_combo)
        end_type_combo.currentIndexChanged.connect(lambda index, row=row_position: self.updateEndCondition(row, index))
        end_edit = QLineEdit()
        self.pattern_table.setCellWidget(row_position, 2, end_edit)
        number_length_edit = QLineEdit()
        number_length_edit.setValidator(QIntValidator())
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
            start = self.pattern_table.cellWidget(row, 0).text()
            end_type = self.pattern_table.cellWidget(row, 1).currentText()
            end = self.pattern_table.cellWidget(row, 2).text()
            number_length = self.pattern_table.cellWidget(row, 3).text()
            pattern_data.append({
                "start": start,
                "end_type": end_type,
                "end": end,
                "number_length": number_length
            })
        return pattern_data

    def updatePattern(self):
        pattern_data = self.getPatternData()
        patterns = []
        for data in pattern_data:
            start = re.escape(data["start"])
            if data["end_type"] == "Single Number":
                end = r'\d'
            elif data["end_type"] == "Multiple Numbers":
                end = r'\d{' + data["number_length"] + '}'
            else:
                end = re.escape(data["end"])
            patterns.append(rf"{start}.*?{end}")
        final_pattern = '|'.join(patterns)
        if self.whole_words.isChecked():
            final_pattern = rf"\b({final_pattern})\b"
        flags = re.DOTALL | (0 if self.case_sensitive.isChecked() else re.IGNORECASE)
        self.final_pattern = re.compile(final_pattern, flags)
        self.pattern_preview.setText(final_pattern)
        self.explanation.setText(f"This pattern will match: {', '.join(patterns)}")

    def testPattern(self):
        self.updatePattern()
        text = self.test_input.toPlainText()
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
            QMessageBox.warning(self, "No Matches", "The pattern did not match any text in the sample.")

    def getPattern(self):
        self.updatePattern()
        return self.final_pattern

class ParametersDialog(QDialog):
    def __init__(self, current_parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Parameters")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        self.parameters = current_parameters.copy()
        self.pattern_data = self.parameters.get("pattern_data", [])
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget)
        options = [
            ("remove_break_lines", "Remove break lines"),
            ("lowercase", "Lowercase"),
            ("strip_html", "Strip HTML tags"),
            ("remove_diacritics", "Diacritic Removal"),
            ("normalize_spacing", "Normalize Spacing"),
            ("remove_stop_words", "Remove Stop Words"),
            ("remove_greek", "Remove Greek letters"),
            ("remove_cyrillic", "Remove Cyrillic script"),
            ("remove_super_sub_script", "Remove superscript and subscript characters")
        ]
        for key, label in options:
            checkbox = QCheckBox(label)
            checkbox.setChecked(self.parameters.get(key, False))
            checkbox.stateChanged.connect(lambda state, k=key: self.parameters.update({k: bool(state)}))
            content_layout.addWidget(checkbox)
        regex_button = QPushButton("Set Pattern")
        regex_button.clicked.connect(self.open_regex_dialog)
        content_layout.addWidget(regex_button)
        self.regex_label = QLabel("Current pattern: " + self.parameters.get("regex_pattern", "None"))
        content_layout.addWidget(self.regex_label)
        char_remove_button = QPushButton("Select Characters to Remove")
        char_remove_button.clicked.connect(self.open_char_selection)
        content_layout.addWidget(char_remove_button)
        self.selected_chars_label = QLabel("Selected items: " + ', '.join(self.parameters.get("chars_to_remove", [])))
        content_layout.addWidget(self.selected_chars_label)
        layout.addWidget(scroll_area)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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
                    self.regex_label.setText("Current pattern: " + pattern.pattern)
                    self.pattern_data = dialog.getPatternData()
                    self.parameters["pattern_data"] = self.pattern_data
                except re.error:
                    QMessageBox.warning(self, "Invalid Pattern", "The entered pattern is invalid.")

    def open_char_selection(self):
        dialog = CharacterSelectionDialog(self.parameters.get("chars_to_remove", []), self)
        if dialog.exec():
            self.parameters["chars_to_remove"] = dialog.get_selected_chars()
            self.selected_chars_label.setText("Selected items: " + ', '.join(self.parameters["chars_to_remove"]))

    def get_parameters(self):
        return self.parameters

class CharacterSelectionDialog(QDialog):
    def __init__(self, current_chars, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Characters to Remove")
        self.setMinimumSize(400, 300)
        layout = QVBoxLayout(self)
        self.selected_chars = list(current_chars)
        input_layout = QHBoxLayout()
        self.char_input = QLineEdit()
        self.char_input.setPlaceholderText("Enter characters or sequences to remove")
        input_layout.addWidget(self.char_input)
        include_button = QPushButton("Include")
        include_button.clicked.connect(self.add_chars)
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
        self.cancel_button.clicked.connect(self.reject)
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
            QMessageBox.warning(self, "Error", f"Error loading a file: {error}")

    def accept(self):
        pass

    def reject(self):
        super().reject()

class DirectoryLoadingWorker(QObject):
    finished = Signal(list)
    def __init__(self, file_manager, directory, signals):
        super().__init__()
        self.file_manager = file_manager
        self.directory = directory
        self.signals = signals

    def run(self):
        new_files = self.file_manager.add_directory(self.directory, self.signals)
        self.finished.emit(new_files)

class ReportWorker(QObject):
    finished = Signal(str, str)
    def __init__(self, files, processed=False, processed_results=None):
        super().__init__()
        self.files = files
        self.processed = processed
        self.processed_results = processed_results

    def run(self):
        total_words = 0
        total_sentences = 0
        all_words = []
        total_size = 0
        if self.processed and self.processed_results:
            for file_path, processed_text in self.processed_results:
                total_size += len(processed_text.encode('utf-8'))
                total_words += len(processed_text.split())
                total_sentences += len(re.split(r'[.!?]+', processed_text.strip()))
                all_words.extend(processed_text.lower().split())
        else:
            for file_path in self.files:
                total_size += os.path.getsize(file_path)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
                    total_words += len(text.split())
                    total_sentences += len(re.split(r'[.!?]+', text.strip()))
                    all_words.extend(text.lower().split())
        avg_words = total_words / len(self.files) if self.files else 0
        avg_sentences = total_sentences / len(self.files) if self.files else 0
        total_size_mb = total_size / (1024 * 1024)
        avg_size_mb = total_size_mb / len(self.files) if self.files else 0
        files_report = f"<h3>{'Processed ' if self.processed else ''}Files Report</h3>"
        files_report += f"<p><b>Total Files:</b> {len(self.files)}</p>"
        files_report += f"<p><b>Total Size:</b> {total_size_mb:.2f} MB</p>"
        files_report += f"<p><b>Average Size per File:</b> {avg_size_mb:.2f} MB</p>"
        files_report += f"<p><b>Earliest File Modification:</b> {time.ctime(min(os.path.getmtime(file) for file in self.files))}</p>"
        files_report += f"<p><b>Latest File Modification:</b> {time.ctime(max(os.path.getmtime(file) for file in self.files))}</p>"
        corpus_report = f"<h3>{'Processed ' if self.processed else ''}Corpus Report</h3>"
        corpus_report += f"<p><b>Word Count:</b> {total_words}</p>"
        corpus_report += f"<p><b>Average Word Count per File:</b> {avg_words:.2f}</p>"
        corpus_report += f"<p><b>Sentences Count:</b> {total_sentences}</p>"
        corpus_report += f"<p><b>Average Sentences Count per File:</b> {avg_sentences:.2f}</p>"
        word_counts = Counter(word for word in all_words if word.isalnum() and len(word) > 1)
        top_25_words = word_counts.most_common(25)
        corpus_report += "<p><b>Word Frequency (Top 25):</b></p>"
        corpus_report += "<ul>"
        for word, count in top_25_words:
            corpus_report += f"<li>{word}: {count}</li>"
        corpus_report += "</ul>"
        self.finished.emit(files_report, corpus_report)

class PreprocessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
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
        self.files_processed = 0
        self.total_size = 0
        self.total_time = 0
        self.processed_size = 0
        self.documentation_window = None
        self.report_thread = QThread()
        self.version = get_version()
        self.init_ui()
        ensure_nltk_data()
        # self.check_for_updates()

    def init_ui(self):
        self.setWindowTitle('CorpuScript')
        self.setGeometry(100, 100, 1200, 800)
        self.setFont(QFont("Roboto", 10))
        self.setWindowIcon(QIcon(resource_path("my_icon.ico")))
        self.create_menu_bar()
        self.create_toolbar()
        self.setup_central_widget()
        self.setup_status_bar()
        self.signals.update_progress.connect(self.update_progress)
        self.signals.result.connect(self.handle_result)
        self.signals.error.connect(self.handle_error)
        self.signals.warning.connect(self.handle_warning)
        self.signals.finished.connect(self.on_worker_finished)
        self.signals.report_ready.connect(self.display_report)
        self.apply_theme()

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.create_action("New", "document-new", "Ctrl+N", "Start a new project", self.start_new_cleaning))
        file_menu.addAction(self.create_action("Open Files", "document-open", "Ctrl+O", "Open files", self.open_file))
        file_menu.addAction(self.create_action("Open Directory", "folder-open", "Ctrl+Shift+O", "Open directory", self.open_directory))
        file_menu.addAction(self.create_action("Save", "document-save", "Ctrl+S", "Save current file", self.save_file))
        file_menu.addSeparator()
        file_menu.addAction(self.create_action("Exit", "application-exit", "Ctrl+Q", "Exit the application", self.close))
        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self.create_action("Undo", "edit-undo", "Ctrl+Z", "Undo last action", self.undo))
        edit_menu.addAction(self.create_action("Redo", "edit-redo", "Ctrl+Y", "Redo last action", self.redo))
        settings_menu = menu_bar.addMenu("&Settings")
        settings_menu.addAction(self.create_action("Toggle Theme", "preferences-desktop-theme", "", "Switch between light and dark theme", self.toggle_theme))
        settings_menu.addAction(self.create_action("Processing Parameters", "preferences-system", "", "Configure processing options", self.open_parameters_dialog))
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self.create_action("About", "help-about", "", "About this application", self.show_about_dialog))
        help_menu.addAction(self.create_action("Documentation", "help-contents", "F1", "View documentation", self.show_documentation))
        # help_menu.addAction(self.create_action("Check for Updates", "system-software-update", "", "Check for updates", self.check_for_updates))

    def create_toolbar(self):
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)
        self.toolbar.addAction(self.create_action("New", "document-new", "Ctrl+Shift+N", "Start a new project", self.start_new_cleaning))
        self.toolbar.addAction(self.create_action("Open Files", "document-open", "Ctrl+O", "Open files", self.open_file))
        self.toolbar.addAction(self.create_action("Open Directory", "folder-open", "Ctrl+Shift+O", "Open directory", self.open_directory))
        self.toolbar.addAction(self.create_action("Save", "document-save", "Ctrl+S", "Save current file", self.save_file))
        self.toolbar.addSeparator()
        process_button = QPushButton("Process Files")
        process_button.setIcon(QIcon.fromTheme("system-run"))
        process_button.setToolTip("Process selected files")
        process_button.clicked.connect(self.process_files)
        self.toolbar.addWidget(process_button)

    def create_action(self, text, icon, shortcut, tooltip, callback):
        action = QAction(QIcon.fromTheme(icon, QIcon()), text, self)
        action.setShortcut(shortcut)
        action.setToolTip(tooltip)
        action.triggered.connect(callback)
        return action

    def setup_central_widget(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        file_list_widget = QWidget()
        file_list_layout = QVBoxLayout(file_list_widget)
        selected_files_label = QLabel("Selected Files:")
        selected_files_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #518FBC;
            margin-bottom: 10px;
        """)
        file_list_layout.addWidget(selected_files_label)
        self.file_list = FileListWidget()
        self.file_list.files_added.connect(lambda files: self.update_report())
        file_list_layout.addWidget(self.file_list)
        report_widget = self.create_report_area()
        left_layout.addWidget(file_list_widget)
        left_layout.addWidget(report_widget)
        text_display = QWidget()
        text_layout = QVBoxLayout(text_display)
        self.text_tabs = QTabWidget()
        self.original_text = QPlainTextEdit()
        self.processed_text = QPlainTextEdit()
        self.text_tabs.addTab(self.original_text, "Original Text")
        self.text_tabs.addTab(self.processed_text, "Processed Text")
        text_layout.addWidget(self.text_tabs)
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search in text...")
        search_button = QPushButton("Search")
        search_button.setIcon(QIcon.fromTheme("edit-find"))
        search_button.clicked.connect(self.search_text)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_button)
        text_layout.addLayout(search_layout)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(text_display)
        main_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)

    def create_report_area(self):
        report_widget = QWidget()
        report_layout = QVBoxLayout(report_widget)
        report_label = QLabel("Summary Report")
        report_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #518FBC;
            margin-bottom: 10px;
        """)
        self.report_tabs = QTabWidget()
        self.files_report_text = QTextEdit()
        self.files_report_text.setReadOnly(True)
        self.corpus_report_text = QTextEdit()
        self.corpus_report_text.setReadOnly(True)
        self.report_tabs.addTab(self.files_report_text, "Files Report")
        self.report_tabs.addTab(self.corpus_report_text, "Corpus Report")
        report_layout.addWidget(report_label)
        report_layout.addWidget(self.report_tabs)
        return report_widget

    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.update_status_bar()

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

    def open_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select files", "", "Text Files (*.txt)")
        if files:
            new_files = self.file_manager.add_files(files)
            self.file_list.addItems(new_files)
            self.update_status_bar()
            self.update_report()

    def open_directory(self):
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
            self.loading_thread.start()

    def on_directory_loading_finished(self, new_files):
        self.loading_thread.quit()
        self.loading_thread.wait()
        self.loading_dialog.close()
        self.file_list.addItems(new_files)
        self.update_status_bar()
        self.update_report()

    def save_file(self):
        if self.current_file and self.processed_text.toPlainText():
            try:
                with open(self.current_file, 'w', encoding='utf-8') as file:
                    file.write(self.processed_text.toPlainText())
                QMessageBox.information(self, 'Save Successful', f"File saved successfully: {self.current_file}")
            except Exception as e:
                logging.error(f"Error saving file {self.current_file}: {str(e)}")
                QMessageBox.warning(self, 'Save Failed', f"Failed to save file: {str(e)}")
        else:
            QMessageBox.warning(self, 'Save Failed', "No processed file to save.")

    def process_files(self):
        logging.debug("PreprocessorGUI: Process files function called")
        if not self.file_manager.get_files():
            QMessageBox.warning(self, "No Files", "Please select files to process.")
            return
        logging.debug(f"PreprocessorGUI: Current processor parameters: {self.processor.get_parameters()}")
        self.processing_dialog = FileLoadingDialog(self, title="Processing Files...")
        self.processing_dialog.show()
        self.signals.update_progress.connect(self.processing_dialog.update_progress)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.file_manager.get_files()))
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

    def handle_result(self, file_path, processed_text, file_size, processing_time):
        self.processed_results.append((file_path, processed_text))
        self.total_time += processing_time
        self.files_processed += 1
        self.processed_size += file_size
        if len(self.file_manager.get_files()) > 0:
            if self.files_processed % 10 == 0 or (self.files_processed / len(self.file_manager.get_files())) * 100 >= self.progress_bar.value() + 1:
                progress = (self.files_processed / len(self.file_manager.get_files())) * 100
                avg_speed = self.processed_size / self.total_time if self.total_time > 0 else 0
                remaining_time = (self.total_size - self.processed_size) / avg_speed if avg_speed > 0 else 0
                self.signals.update_progress.emit(self.files_processed, len(self.file_manager.get_files()), remaining_time, None)

    def handle_error(self, file_path, error):
        self.errors.append((file_path, error))
        self.files_processed += 1
        self.signals.update_progress.emit(self.files_processed, len(self.file_manager.get_files()), 0, error)

    def handle_warning(self, file_path, warning):
        self.warnings.append((file_path, warning))
        self.files_processed += 1
        self.signals.update_progress.emit(self.files_processed, len(self.file_manager.get_files()), 0, None)

    def update_progress(self, current, total, time_remaining, error):
        self.files_processed = current
        progress = (self.files_processed / total) * 100 if total > 0 else 0
        self.progress_bar.setValue(int(progress))
        avg_speed = self.processed_size / self.total_time if self.total_time > 0 else 0
        self.status_bar.showMessage(f"Progress: {progress:.2f}% | Avg. Speed: {avg_speed:.2f} B/s | Est. Remaining Time: {time_remaining:.2f}s")

    def on_worker_finished(self):
        logging.debug("Worker finished")
        if self.files_processed == len(self.file_manager.get_files()):
            self.signals.processing_complete.emit(self.processed_results, self.warnings)
            if self.errors:
                error_msg = "\n".join([f"{file}: {error}" for file, error in self.errors])
                QMessageBox.warning(self, "Processing Errors", f"Errors occurred during processing:\n\n{error_msg}")
            self.update_status_bar()
            self.processing_dialog.close()
            self.display_results(self.processed_results, self.warnings)
            self.update_report(processed=True, processed_results=self.processed_results)

    def display_results(self, results, warnings):
        logging.debug(f"Displaying results for {len(results)} files")
        if results:
            self.original_text.clear()
            self.processed_text.clear()
            for file_path, processed_text in results:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    original_text = file.read()
                self.original_text.appendPlainText(f"File: {file_path}\n\n{original_text}\n\n")
                self.processed_text.appendPlainText(f"File: {file_path}\n\n{processed_text}\n\n")
            self.current_file = results[0][0]
        if warnings:
            warning_msg = "\n".join([f"{file}: {warning}" for file, warning in warnings])
            QMessageBox.information(self, "Processing Warnings",
                                    f"The following files were skipped or had issues:\n\n{warning_msg}")

    def start_new_cleaning(self):
        self.file_manager.clear_files()
        self.file_list.clear()
        self.original_text.clear()
        self.processed_text.clear()
        self.current_file = None
        self.corpus_name = "Untitled Corpus"
        self.update_status_bar()
        self.update_report()

    def open_parameters_dialog(self):
        dialog = ParametersDialog(self.processor.get_parameters(), self)
        if dialog.exec():
            new_parameters = dialog.get_parameters()
            logging.debug(f"New parameters set: {new_parameters}")
            self.processor.set_parameters(new_parameters)
            self.processor.update_pipeline()
            logging.debug(f"Updated pipeline modules: {[type(module).__name__ for module in self.processor.pipeline.modules]}")

    def show_about_dialog(self):
        QMessageBox.about(self, "About", f"CorpuScript\nVersion {self.version}\n\nDeveloped by Jhonatan Lopes")

    def show_documentation(self):
        if self.documentation_window is None:
            readme_file = resource_path("README.md")
            try:
                with open(readme_file, "r", encoding="utf-8") as f:
                    readme_content = f.read()
                self.documentation_window = QTextEdit()
                self.documentation_window.setWindowTitle("Documentation")
                self.documentation_window.resize(800, 600)
                font = QFont("Roboto", 12)
                self.documentation_window.setFont(font)
                self.documentation_window.setMarkdown(readme_content)
                self.documentation_window.setReadOnly(True)
                self.documentation_window.setWordWrapMode(QTextOption.WordWrap)
            except FileNotFoundError:
                QMessageBox.warning(self, "Documentation Not Found", f"The README.md file could not be found: {readme_file}")
                return
            except Exception as e:
                QMessageBox.warning(self, "Error", f"An error occurred while reading the documentation: {str(e)}")
                return
        self.documentation_window.show()

    def search_text(self):
        search_term = self.search_input.text()
        if search_term:
            active_text_edit = self.text_tabs.currentWidget()
            if isinstance(active_text_edit, QPlainTextEdit):
                found = active_text_edit.find(search_term)
                if not found:
                    QMessageBox.information(self, "Not Found",
                                            f"The text '{search_term}' was not found.")

    def undo(self):
        active_text_edit = self.text_tabs.currentWidget()
        if isinstance(active_text_edit, QPlainTextEdit):
            active_text_edit.undo()

    def redo(self):
        active_text_edit = self.text_tabs.currentWidget()
        if isinstance(active_text_edit, QPlainTextEdit):
            active_text_edit.redo()

    def update_report(self, processed=False, processed_results=None):
        if not self.file_manager.get_files():
            self.files_report_text.clear()
            self.corpus_report_text.clear()
            return
        if self.report_thread.isRunning():
            self.report_thread.quit()
            self.report_thread.wait()
        self.report_worker = ReportWorker(self.file_manager.get_files(), processed, processed_results)
        self.report_worker.moveToThread(self.report_thread)
        self.report_thread.started.connect(self.report_worker.run)
        self.report_worker.finished.connect(self.display_report)
        self.report_thread.start()

    def display_report(self, files_report, corpus_report):
        self.files_report_text.setHtml(files_report)
        self.corpus_report_text.setHtml(corpus_report)

    # def check_for_updates(self):
    #     self.update_checker = UpdateChecker(self.version)
    #     self.update_checker.update_available.connect(self.show_update_dialog)
    #     self.update_checker.error.connect(lambda e: logging.error(f"Update check failed: {e}"))
    #     self.update_checker.start()

    # def show_update_dialog(self, latest_version):
    #     reply = QMessageBox.question(self, "Update Available",
    #                                 f"A new version ({latest_version}) is available. Would you like to download it?",
    #                                 QMessageBox.Yes | QMessageBox.No)
    #     if reply == QMessageBox.Yes:
    #         QDesktopServices.openUrl(QUrl("https://github.com/YourUsername/CorpuScript/releases/latest"))

def main():
    app = QApplication(sys.argv)
    window = PreprocessorGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    main()