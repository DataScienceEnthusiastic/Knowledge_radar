# Apache Lucene Inverted Index Structure

Lucene indexes text by tokenizing documents into terms and constructing an inverted index. The inverted index maps each term to a posting list containing document IDs, term frequencies, and token positions. Posting lists are compressed using bit-packing and Frame of Reference (FOR) compression algorithms. Segments are immutable on disk; updates append new documents and mark deleted document IDs in a bitset.
