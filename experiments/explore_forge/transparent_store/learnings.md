# Learned rules (self-modifiable supplemental prompt)

- If query contains multiple named entities and asks for their connection, use entity_relation_connection skill with hybrid and fielded retrievers combined by fuse.
- If query focuses on attributes or heritage of a single entity, use entity_attribute_retrieval skill with fielded and hybrid retrievers combined by fuse.
- If query involves events and temporal comparison, use event_comparison_retrieval skill with fielded retriever.
- If query requests titles and authors of literary works with specific phrases, use novel_title_author_retrieval skill with fielded retriever.
- If query involves products linked to companies or origins, use product_origin_retrieval skill with fielded retriever.
- If query involves institutional status or programs, use institution_status_retrieval skill with fielded retriever.