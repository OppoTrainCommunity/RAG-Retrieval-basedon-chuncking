"""
Template Parser Module
======================

Loads prompt templates from organized files by language and group.
Supports multiple languages (en, ar) with automatic fallback to default.

Usage:
    from rag.template_parser import template_parser

    # Get a prompt template string
    prompt = template_parser.get("rag", "system_prompt")

    # Get with variable substitution
    prompt = template_parser.get("rag", "document_prompt", {"doc_num": 1, "chunk_text": "..."})

    # Switch language at runtime
    template_parser.set_language("ar")
"""

import os
import importlib
from typing import Optional
from .logging_utils import get_logger

logger = get_logger(__name__)


class TemplateParser:
    """
    Manages prompt templates organized by language and group.

    Directory structure:
        rag/templates/
        ├── en/
        │   ├── rag.py          # RAG prompts
        │   ├── summary.py      # CV summary prompts
        │   └── compare.py      # Candidate comparison prompts
        └── ar/
            ├── rag.py
            ├── summary.py
            └── compare.py
    """

    def __init__(self, language: str = "en", default_language: str = "en"):
        self.templates_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates"
        )
        self.default_language = default_language
        self.language = default_language
        self.set_language(language)

    def set_language(self, language: str):
        """
        Set the active language. Falls back to default if not available.

        Args:
            language: Language code (e.g., 'en', 'ar')
        """
        if not language:
            self.language = self.default_language
            return

        language_path = os.path.join(self.templates_dir, language)
        if os.path.exists(language_path):
            self.language = language
            logger.info(f"Template language set to: {language}")
        else:
            logger.warning(
                f"Language '{language}' not found at {language_path}. "
                f"Falling back to '{self.default_language}'"
            )
            self.language = self.default_language

    def get(self, group: str, key: str, vars: Optional[dict] = None) -> Optional[str]:
        """
        Get a prompt template by group and key, with optional variable substitution.

        Args:
            group: Template group name (e.g., 'rag', 'summary', 'compare')
            key: Template key/variable name within the group
            vars: Optional dict of variables to substitute using str.format()

        Returns:
            The template string with variables substituted, or None if not found.
        """
        if not group or not key:
            return None

        vars = vars or {}

        # Try current language first
        template = self._load_template(self.language, group, key)

        # Fallback to default language
        if template is None and self.language != self.default_language:
            logger.debug(
                f"Template '{group}.{key}' not found in '{self.language}', "
                f"falling back to '{self.default_language}'"
            )
            template = self._load_template(self.default_language, group, key)

        if template is None:
            logger.warning(f"Template not found: {group}.{key}")
            return None

        # Substitute variables if provided
        if vars:
            try:
                return template.substitute(vars) if hasattr(template, 'substitute') else template.format(**vars)
            except (KeyError, ValueError) as e:
                logger.error(f"Variable substitution failed for {group}.{key}: {e}")
                return template.template if hasattr(template, 'template') else str(template)

        # Return raw template string (with placeholders intact)
        if hasattr(template, 'template'):
            return template.template
        return template if isinstance(template, str) else str(template)

    def _load_template(self, language: str, group: str, key: str) -> Optional[str]:
        """
        Load a template from the module system.

        Args:
            language: Language code
            group: Template group (file name without .py)
            key: Attribute name in the module

        Returns:
            Template string or Template object, or None if not found.
        """
        module_path = f"rag.templates.{language}.{group}"

        try:
            module = importlib.import_module(module_path)
            if hasattr(module, key):
                return getattr(module, key)
            else:
                logger.debug(f"Key '{key}' not found in module '{module_path}'")
                return None
        except ModuleNotFoundError:
            logger.debug(f"Module not found: {module_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading template {module_path}.{key}: {e}")
            return None

    def get_available_languages(self) -> list:
        """Return list of available language codes."""
        if not os.path.exists(self.templates_dir):
            return []
        return [
            d for d in os.listdir(self.templates_dir)
            if os.path.isdir(os.path.join(self.templates_dir, d))
            and not d.startswith("_")
        ]

    def get_available_groups(self, language: str = None) -> list:
        """Return list of available template groups for a language."""
        lang = language or self.language
        lang_path = os.path.join(self.templates_dir, lang)
        if not os.path.exists(lang_path):
            return []
        return [
            f[:-3] for f in os.listdir(lang_path)
            if f.endswith(".py") and not f.startswith("_")
        ]


# ── Global instance ────────────────────────────────────────────
# Language can be overridden via config or at runtime
template_parser = TemplateParser(language="en")
