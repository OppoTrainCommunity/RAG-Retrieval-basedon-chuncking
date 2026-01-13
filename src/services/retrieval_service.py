class RetrievalService:
    def __init__(self, persist_dir, embedding_model):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model

    def build_index(
        self,
        cvs_json_path,
        collection_name,
        paragraph_config: dict,
        rebuild=False,
    ):
        from src.preprocessors import CVDataPreprocessor
        from src.chunkers import ParagraphChunker
        from src.indexers import ChromaIndexer

        pre = CVDataPreprocessor()
        texts, metas = pre.prepare_data(cvs_json_path)

        chunker = ParagraphChunker(**paragraph_config)
        chunks, chunk_metas, ids = chunker.chunk_texts(texts, metas)

        indexer = ChromaIndexer(
            collection_name=collection_name,
            embedding_model=self.embedding_model,
            persist_directory=self.persist_dir,
            client_type="persistent",
        )

        if rebuild:
            indexer.delete_collection()

        indexer.add_chunks(chunks, chunk_metas, ids)

    def search(self, collection_name, query, k=10, where=None):
        from src.indexers import ChromaIndexer

        indexer = ChromaIndexer(
            collection_name=collection_name,
            embedding_model=self.embedding_model,
            persist_directory=self.persist_dir,
            client_type="persistent",
        )
        return indexer.search(query, n_results=k, where=where)
