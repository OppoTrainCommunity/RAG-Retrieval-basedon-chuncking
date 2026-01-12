"""
ChromaDB indexer implementation.

This module provides functionality to index text chunks into ChromaDB
vector database for semantic search and retrieval.
"""
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
from .base import BaseIndexer


class ChromaIndexer(BaseIndexer):
    """
    ChromaDB indexer for storing and searching text chunks.
    
    This class handles the indexing of text chunks into ChromaDB, a vector
    database optimized for semantic search using embeddings.
    
    Attributes:
        collection_name (str): Name of the ChromaDB collection
        client: ChromaDB client instance
        embedding_function: Function to generate embeddings
        collection: ChromaDB collection instance
        
    Methods:
        add_chunks: Add text chunks to ChromaDB
        search: Search for similar chunks using semantic similarity
        get_collection_stats: Get collection metadata and statistics
        delete_collection: Remove the collection from ChromaDB
    """
    
    def __init__(self, 
                 collection_name: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: Optional[str] = "./chroma_db",
                 client_type: str = "persistent"):
        """
        Initialize the ChromaDB indexer.
        
        Args:
            collection_name (str): Name for the ChromaDB collection
            embedding_model (str, optional): Name of the sentence transformer model
                for embeddings. Defaults to "all-MiniLM-L6-v2".
            persist_directory (Optional[str], optional): Directory to persist
                the database. Defaults to "./chroma_db".
            client_type (str, optional): Type of ChromaDB client to use.
                Options: "persistent" or "ephemeral". Defaults to "persistent".
                
        Raises:
            ValueError: If client_type is not "persistent" or "ephemeral"
            
        Examples:
            >>> indexer = ChromaIndexer("cv_chunks", embedding_model="all-MiniLM-L6-v2")
            >>> indexer.add_chunks(chunks, metadatas, ids)
        """
        super().__init__(collection_name)
        
        # Initialize ChromaDB client
        if client_type == "persistent":
            self.client = chromadb.PersistentClient(path=persist_directory)
        elif client_type == "ephemeral":
            self.client = chromadb.EphemeralClient()
        else:
            raise ValueError(f"Invalid client_type: {client_type}. Use 'persistent' or 'ephemeral'")
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": f"Collection for {collection_name}"}
        )
        
        print(f"ChromaDB collection '{collection_name}' initialized")
    
    def add_chunks(self, chunks: List[str], metadatas: List[Dict], 
                   chunk_ids: List[str], batch_size: int = 100) -> None:
        """
        Add text chunks to ChromaDB collection.
        
        Adds chunks in batches to avoid memory issues with large datasets.
        
        Args:
            chunks (List[str]): List of text chunks to index
            metadatas (List[Dict]): List of metadata dictionaries for each chunk
            chunk_ids (List[str]): List of unique IDs for each chunk
            batch_size (int, optional): Number of chunks to add per batch.
                Defaults to 100.
                
        Returns:
            None
            
        Raises:
            ValueError: If lengths of chunks, metadatas, and chunk_ids don't match
            
        Examples:
            >>> indexer = ChromaIndexer("my_collection")
            >>> indexer.add_chunks(
            ...     chunks=["text 1", "text 2"],
            ...     metadatas=[{"id": 1}, {"id": 2}],
            ...     chunk_ids=["chunk_1", "chunk_2"]
            ... )
            
        Note:
            Large batches may cause memory issues. Adjust batch_size accordingly.
        """
        # Validate input lengths
        if not (len(chunks) == len(metadatas) == len(chunk_ids)):
            raise ValueError(
                f"Length mismatch: chunks({len(chunks)}), "
                f"metadatas({len(metadatas)}), chunk_ids({len(chunk_ids)})"
            )
        
        # Add chunks in batches
        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            
            self.collection.add(
                documents=chunks[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=chunk_ids[i:batch_end]
            )
            
            print(f"Added chunks {i+1}-{batch_end} of {total_chunks}")
        
        print(f"Successfully added {total_chunks} chunks to '{self.collection_name}'")
    
    def search(self, query: str, n_results: int = 5, 
               where: Optional[Dict] = None,
               where_document: Optional[Dict] = None) -> Dict:
        """
        Search for similar chunks using semantic similarity.
        
        Performs vector similarity search to find chunks most relevant to
        the query based on embedding similarity.
        
        Args:
            query (str): Search query text
            n_results (int, optional): Number of results to return. Defaults to 5.
            where (Optional[Dict], optional): Metadata filter conditions.
                Example: {"doc_id": "doc1"}. Defaults to None.
            where_document (Optional[Dict], optional): Document content filter.
                Example: {"$contains": "keyword"}. Defaults to None.
                
        Returns:
            Dict: Dictionary containing:
                - 'ids': List of chunk IDs
                - 'documents': List of chunk texts
                - 'metadatas': List of metadata dictionaries
                - 'distances': List of similarity distances
                
        Examples:
            >>> results = indexer.search("machine learning", n_results=3)
            >>> for doc, meta, dist in zip(results['documents'][0], 
            ...                            results['metadatas'][0],
            ...                            results['distances'][0]):
            ...     print(f"Distance: {dist}, Text: {doc[:100]}")
            
            >>> # Search with metadata filter
            >>> results = indexer.search(
            ...     "python programming",
            ...     where={"doc_id": "cv_123"}
            ... )
            
        Note:
            Lower distances indicate higher similarity
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the ChromaDB collection.
        
        Returns:
            Dict: Dictionary containing:
                - 'name': Collection name
                - 'count': Number of chunks in collection
                - 'metadata': Collection metadata
                
        Examples:
            >>> stats = indexer.get_collection_stats()
            >>> print(f"Collection has {stats['count']} chunks")
        """
        count = self.collection.count()
        metadata = self.collection.metadata
        
        return {
            'name': self.collection_name,
            'count': count,
            'metadata': metadata
        }
    
    def delete_collection(self) -> None:
        """
        Delete the ChromaDB collection.
        
        Permanently removes the collection and all its data from ChromaDB.
        
        Returns:
            None
            
        Warning:
            This operation is irreversible. All data in the collection will be lost.
            
        Examples:
            >>> indexer = ChromaIndexer("temp_collection")
            >>> indexer.add_chunks(chunks, metas, ids)
            >>> indexer.delete_collection()  # All data is now deleted
        """
        self.client.delete_collection(name=self.collection_name)
        print(f"Collection '{self.collection_name}' deleted successfully")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id (str): Unique ID of the chunk to retrieve
            
        Returns:
            Optional[Dict]: Dictionary containing chunk data, or None if not found:
                - 'id': Chunk ID
                - 'document': Chunk text
                - 'metadata': Chunk metadata
                
        Examples:
            >>> chunk = indexer.get_chunk_by_id("doc1_sem_0")
            >>> if chunk:
            ...     print(chunk['document'])
        """
        try:
            result = self.collection.get(ids=[chunk_id])
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except Exception as e:
            print(f"Error retrieving chunk {chunk_id}: {e}")
        return None
    
    def update_metadata(self, chunk_ids: List[str], metadatas: List[Dict]) -> None:
        """
        Update metadata for existing chunks.
        
        Args:
            chunk_ids (List[str]): List of chunk IDs to update
            metadatas (List[Dict]): List of new metadata dictionaries
            
        Returns:
            None
            
        Raises:
            ValueError: If lengths of chunk_ids and metadatas don't match
            
        Examples:
            >>> indexer.update_metadata(
            ...     chunk_ids=["chunk_1", "chunk_2"],
            ...     metadatas=[{"priority": "high"}, {"priority": "low"}]
            ... )
        """
        if len(chunk_ids) != len(metadatas):
            raise ValueError("chunk_ids and metadatas must have same length")
        
        self.collection.update(
            ids=chunk_ids,
            metadatas=metadatas
        )
        print(f"Updated metadata for {len(chunk_ids)} chunks")
