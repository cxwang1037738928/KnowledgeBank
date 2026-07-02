# KnowledgeBank
A data crawler that takes in massive samples of scanned PDFs, calls an AI API to parse a small section of the PDFs, then runs LORA training on an open source model to specialize it in parsing PDFs. The, use the parsed info, as well as other techniques like KAG, TF-IDF, and RAG to generate prompts for ML models to answer questions abou the data. 
