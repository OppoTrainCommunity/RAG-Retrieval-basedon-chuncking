"""
Semantic chunking for CVs.
Chunks CVs by sections (Summary, Experience, Education, Skills, etc.)
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Section patterns for CV parsing
SECTION_PATTERNS = {
    "summary": [
        r"(?i)^(?:professional\s+)?summary\s*[:|\n]",
        r"(?i)^profile\s*[:|\n]",
        r"(?i)^about\s+me\s*[:|\n]",
        r"(?i)^objective\s*[:|\n]",
        r"(?i)^career\s+(?:summary|objective)\s*[:|\n]",
    ],
    "experience": [
        r"(?i)^(?:work\s+)?experience\s*[:|\n]",
        r"(?i)^employment\s+(?:history)?\s*[:|\n]",
        r"(?i)^work\s+history\s*[:|\n]",
        r"(?i)^professional\s+experience\s*[:|\n]",
        r"(?i)^career\s+history\s*[:|\n]",
    ],
    "education": [
        r"(?i)^education\s*[:|\n]",
        r"(?i)^academic\s+(?:background|qualifications)\s*[:|\n]",
        r"(?i)^qualifications\s*[:|\n]",
    ],
    "skills": [
        r"(?i)^(?:technical\s+)?skills\s*[:|\n]",
        r"(?i)^core\s+competencies\s*[:|\n]",
        r"(?i)^expertise\s*[:|\n]",
        r"(?i)^technical\s+proficiencies\s*[:|\n]",
        r"(?i)^key\s+skills\s*[:|\n]",
    ],
    "projects": [
        r"(?i)^projects?\s*[:|\n]",
        r"(?i)^personal\s+projects?\s*[:|\n]",
        r"(?i)^portfolio\s*[:|\n]",
        r"(?i)^side\s+projects?\s*[:|\n]",
    ],
    "certifications": [
        r"(?i)^certifications?\s*[:|\n]",
        r"(?i)^licenses?\s+(?:and\s+)?certifications?\s*[:|\n]",
        r"(?i)^professional\s+certifications?\s*[:|\n]",
        r"(?i)^credentials?\s*[:|\n]",
    ],
    "publications": [
        r"(?i)^publications?\s*[:|\n]",
        r"(?i)^research\s*[:|\n]",
        r"(?i)^papers?\s*[:|\n]",
        r"(?i)^articles?\s*[:|\n]",
    ],
    "awards": [
        r"(?i)^awards?\s*[:|\n]",
        r"(?i)^honors?\s*[:|\n]",
        r"(?i)^achievements?\s*[:|\n]",
        r"(?i)^recognition\s*[:|\n]",
        r"(?i)^awards?\s+(?:and\s+)?honors?\s*[:|\n]",
    ],
    "header": [
        r"^[A-Z][A-Z\s]+$",  # All caps name at start
    ],
}


@dataclass
class CVChunk:
    """Represents a chunk of a CV."""
    chunk_id: str
    candidate_id: str
    section_name: str
    chunk_text: str
    chunk_index: int
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "candidate_id": self.candidate_id,
            "section_name": self.section_name,
            "chunk_text": self.chunk_text,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


def generate_chunk_id(candidate_id: str, section_name: str, index: int) -> str:
    """
    Generate a deterministic chunk ID.
    
    Args:
        candidate_id: ID of the candidate.
        section_name: Name of the CV section.
        index: Index of the chunk within the section.
    
    Returns:
        Unique chunk ID.
    """
    content = f"{candidate_id}|{section_name}|{index}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def detect_section(line: str) -> Optional[str]:
    """
    Detect if a line is a section header.
    
    Args:
        line: Text line to check.
    
    Returns:
        Section name if detected, None otherwise.
    """
    line = line.strip()
    if not line:
        return None
    
    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, line):
                return section_name
    
    return None


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split CV text into sections.
    
    Args:
        text: Full CV text.
    
    Returns:
        List of (section_name, section_text) tuples.
    """
    lines = text.split('\n')
    sections = []
    current_section = "header"
    current_content = []
    
    for line in lines:
        detected = detect_section(line)
        
        if detected and detected != current_section:
            # Save current section
            if current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    sections.append((current_section, content))
            
            # Start new section
            current_section = detected
            current_content = [line]
        else:
            current_content.append(line)
    
    # Don't forget the last section
    if current_content:
        content = '\n'.join(current_content).strip()
        if content:
            sections.append((current_section, content))
    
    return sections


def extract_experience_roles(experience_text: str) -> List[Tuple[str, str]]:
    """
    Extract individual roles from experience section.
    
    Args:
        experience_text: Text of the experience section.
    
    Returns:
        List of (role_title, role_text) tuples.
    """
    # Pattern to detect role entries (Title | Company | Date)
    role_pattern = r"(?m)^([A-Za-z\s]+)\s*[|@]\s*([A-Za-z\s\.,]+)\s*[|]\s*(\d{4}\s*[-–]\s*(?:\d{4}|Present))"
    
    matches = list(re.finditer(role_pattern, experience_text))
    
    if not matches:
        # Fallback: return whole section as one chunk
        return [("experience", experience_text)]
    
    roles = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(experience_text)
        
        role_title = match.group(1).strip()
        company = match.group(2).strip()
        role_text = experience_text[start:end].strip()
        
        roles.append((f"{role_title} @ {company}", role_text))
    
    return roles


class CVChunker:
    """
    Semantic chunker for CVs.
    Splits CVs by sections and optionally by roles within experience.
    """
    
    def __init__(
        self,
        split_experience_roles: bool = True,
        min_chunk_length: int = 50,
        max_chunk_length: int = 2000,
    ):
        """
        Initialize the CV chunker.
        
        Args:
            split_experience_roles: Whether to split experience into individual roles.
            min_chunk_length: Minimum chunk length in characters.
            max_chunk_length: Maximum chunk length in characters.
        """
        self.split_experience_roles = split_experience_roles
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
    
    def chunk_cv(
        self,
        candidate_id: str,
        raw_text: str,
        metadata: Optional[Dict] = None,
    ) -> List[CVChunk]:
        """
        Chunk a single CV into semantic sections.
        
        Args:
            candidate_id: ID of the candidate.
            raw_text: Full CV text.
            metadata: Optional metadata to include with each chunk.
        
        Returns:
            List of CVChunk objects.
        """
        if metadata is None:
            metadata = {}
        
        chunks = []
        sections = split_into_sections(raw_text)
        
        chunk_counter = 0
        
        for section_name, section_text in sections:
            # Handle experience section specially
            if section_name == "experience" and self.split_experience_roles:
                roles = extract_experience_roles(section_text)
                for role_title, role_text in roles:
                    if len(role_text) >= self.min_chunk_length:
                        chunk = CVChunk(
                            chunk_id=generate_chunk_id(candidate_id, section_name, chunk_counter),
                            candidate_id=candidate_id,
                            section_name=section_name,
                            chunk_text=role_text,
                            chunk_index=chunk_counter,
                            metadata={
                                **metadata,
                                "role_title": role_title,
                            },
                        )
                        chunks.append(chunk)
                        chunk_counter += 1
            else:
                # Split long sections
                if len(section_text) > self.max_chunk_length:
                    sub_chunks = self._split_long_section(section_text)
                else:
                    sub_chunks = [section_text]
                
                for sub_text in sub_chunks:
                    if len(sub_text) >= self.min_chunk_length:
                        chunk = CVChunk(
                            chunk_id=generate_chunk_id(candidate_id, section_name, chunk_counter),
                            candidate_id=candidate_id,
                            section_name=section_name,
                            chunk_text=sub_text,
                            chunk_index=chunk_counter,
                            metadata=metadata.copy(),
                        )
                        chunks.append(chunk)
                        chunk_counter += 1
        
        logger.debug(f"Created {len(chunks)} chunks for candidate {candidate_id}")
        return chunks
    
    def _split_long_section(self, text: str) -> List[str]:
        """
        Split a long section into smaller chunks.
        
        Args:
            text: Text to split.
        
        Returns:
            List of text chunks.
        """
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_length + len(para) > self.max_chunk_length:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = len(para)
            else:
                current_chunk.append(para)
                current_length += len(para)
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks


def chunk_dataframe(
    df: pd.DataFrame,
    candidate_id_col: str = "candidate_id",
    text_col: str = "raw_text",
    metadata_cols: Optional[List[str]] = None,
    chunker: Optional[CVChunker] = None,
) -> pd.DataFrame:
    """
    Chunk all CVs in a DataFrame.
    
    Args:
        df: DataFrame with CV data.
        candidate_id_col: Name of the candidate ID column.
        text_col: Name of the raw text column.
        metadata_cols: List of columns to include as metadata.
        chunker: CVChunker instance (creates default if None).
    
    Returns:
        DataFrame with chunk data.
    """
    if chunker is None:
        chunker = CVChunker()
    
    if metadata_cols is None:
        # Use all columns except the text column as metadata
        metadata_cols = [c for c in df.columns if c not in [candidate_id_col, text_col]]
    
    all_chunks = []
    
    for _, row in df.iterrows():
        candidate_id = str(row[candidate_id_col])
        raw_text = row[text_col]
        
        # Extract metadata
        metadata = {col: row[col] for col in metadata_cols if col in row.index}
        
        # Chunk the CV
        chunks = chunker.chunk_cv(candidate_id, raw_text, metadata)
        
        for chunk in chunks:
            all_chunks.append(chunk.to_dict())
    
    chunks_df = pd.DataFrame(all_chunks)
    
    logger.info(f"Created {len(chunks_df)} total chunks from {len(df)} CVs")
    
    return chunks_df


def save_chunks(
    chunks_df: pd.DataFrame,
    output_path: str = "./data/processed/chunks.parquet",
) -> None:
    """
    Save chunks DataFrame to a Parquet file.
    
    Args:
        chunks_df: DataFrame with chunk data.
        output_path: Path to save the Parquet file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    chunks_df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {len(chunks_df)} chunks to {output_path}")


def load_chunks(chunks_path: str = "./data/processed/chunks.parquet") -> pd.DataFrame:
    """
    Load chunks from a Parquet file.
    
    Args:
        chunks_path: Path to the Parquet file.
    
    Returns:
        DataFrame with chunk data.
    """
    return pd.read_parquet(chunks_path)
