# Use local Chroma for the initial partner knowledge index

The first release will persist the partner knowledge index in local Chroma. This provides a self-contained vector store for the local document source while OCI deployment and document storage are still undecided; retrieval code should keep the store boundary replaceable for a future OCI-hosted alternative.
