export const WEBSEM_FORMAT_VERSION = "1.0" as const;

export interface Section {
  text: string;
  heading?: string;
  anchor?: string;
}

export interface BuildDocument {
  id: string;
  title: string;
  href: string;
  text: string;
  sections?: Section[];
}

export interface ChunkRecord {
  document: string;
  title: string;
  href: string;
  snippet: string;
  heading?: string;
  anchor?: string;
}

export interface ArtifactFiles {
  chunks: string;
  docs: string;
  tokens: string;
  scales: string;
  vocab: string;
}

export interface Manifest {
  websem_version: typeof WEBSEM_FORMAT_VERSION;
  model_id: string;
  dims: number;
  vocab_size: number;
  n_chunks: number;
  chunk_size: number;
  chunk_overlap: number;
  title_prefix: boolean;
  chunker_version: string;
  doc_scale: 127;
  files: ArtifactFiles;
}

export interface SearchResult extends ChunkRecord {
  score: number;
}

export interface LoadedArtifacts {
  manifest: Manifest;
  chunks: ChunkRecord[];
  docs: Int8Array;
  tokens: Int8Array;
  scales: Float32Array;
  vocabulary: string[];
}

export interface SearchOptions {
  limit?: number;
  minScore?: number;
  specificTermHeuristic?: boolean;
}

export interface HybridSearchOptions extends SearchOptions {
  rrfK?: number;
  semanticWeight?: number;
  keywordWeight?: number;
}
