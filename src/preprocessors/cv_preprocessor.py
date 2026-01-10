"""
CV data preprocessor implementation.

This module provides a concrete preprocessor for handling CV (resume) data
stored in JSON format.
"""

import json
from typing import List, Dict, Tuple
from .base import BasePreprocessor


class CVDataPreprocessor(BasePreprocessor):
    """
    Preprocessor for CV (resume) JSON data.
    
    This class handles loading and preprocessing of CV data from JSON files,
    extracting text content and associated metadata.
    
    Attributes:
        None
        
    Methods:
        prepare_data: Load CV data from JSON file
    """
    
    def prepare_data(self, data_path: str) -> Tuple[List[str], List[Dict]]:
        """
        Load CV data from JSON file.
        
        Reads a JSON file containing CV documents and extracts the text content
        along with metadata for each document.
        
        Args:
            data_path (str): Path to the JSON file containing CV data
            
        Returns:
            Tuple[List[str], List[Dict]]: A tuple containing:
                - List of cleaned CV text documents
                - List of metadata dictionaries with keys:
                    - 'doc_id': Unique document identifier
                    - 'file_name': Original file name
                    - 'source': Source of the document
                    
        Raises:
            FileNotFoundError: If the data_path does not exist
            json.JSONDecodeError: If the file is not valid JSON
            KeyError: If required fields are missing from JSON
            
        Examples:
            >>> preprocessor = CVDataPreprocessor()
            >>> texts, metas = preprocessor.prepare_data("data/CVs.json")
            >>> len(texts) == len(metas)
            True
        """
        with open(data_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        texts = []
        metadatas = []
        
        for item in docs:
            text = self.clean_pdf_text(item["text"])
            texts.append(text)
            metadatas.append({
                "doc_id": item["id"],
                "file_name": item["file_name"],
                "source": item.get("source", "unknown")
            })
        
        return texts, metadatas

