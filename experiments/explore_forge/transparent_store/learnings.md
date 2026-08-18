# Learned rules (self-modifiable supplemental prompt)

- If query contains multiple named entities and asks for their connection, use entity_relation_connection skill with hybrid and fielded retrievers combined by fuse.
- If query focuses on attributes or heritage of a single entity, use entity_attribute_retrieval skill with fielded and hybrid retrievers combined by fuse.
- If query involves events and temporal comparison, use event_comparison_retrieval skill with fielded retriever.
- If query requests titles and authors of literary works with specific phrases, use novel_title_author_retrieval skill with fielded retriever.
- If query involves products linked to companies or origins, use product_origin_retrieval skill with fielded retriever.
- If query involves institutional status or programs, use institution_status_retrieval skill with fielded retriever.
- If question involves biological taxonomy and families, use skill 'fielded_family_species_retrieval'.
- If question involves geographic locations or administrative divisions, use skill 'fielded_location_entity_retrieval'.
- If question involves people filtered by birth years, number of works, or titles, use skill 'hybrid_biographical_filtering'.
- If question involves professional roles or attributes tied to birth years or other qualifiers, use skill 'fielded_professional_role_retrieval'.
- If question involves sports clubs by country or league and their achievements, use skill 'hybrid_sports_club_retrieval'.
- If question involves awards or nominations for named entities, use skill 'fielded_awards_nominations_retrieval'.