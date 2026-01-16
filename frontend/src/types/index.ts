export type BlurType = 'gaussian' | 'mosaic' | 'blackout';

export type ProcessStatus = 'uploaded' | 'processing' | 'analyzing' | 'completed' | 'failed';

export interface Resolution {
  width: number;
  height: number;
}

export interface VideoUploadResponse {
  video_id: string;
  filename: string;
  duration: number;
  resolution: Resolution;
  status: ProcessStatus;
}

export interface ReferenceUploadResponse {
  reference_id: string;
  face_count: number;
  embeddings_generated: boolean;
}

export interface FaceAppearance {
  start: number;
  end: number;
  bbox: number[];
}

export interface DetectedFace {
  face_id: string;
  thumbnail: string;
  first_appearance: number;
  appearances: FaceAppearance[];
  similarity_to_reference: number;
  is_reference: boolean;
  blur_enabled: boolean;
}

export interface AnalysisResult {
  analysis_id: string;
  faces: DetectedFace[];
  total_faces: number;
  reference_matches: number;
  status: ProcessStatus;
}

export interface BlurSettings {
  type: BlurType;
  intensity: number;
  face_ids: string[];
}

export interface AnalysisProgress {
  progress: number;
  faces_detected: number;
  frames_processed: number;
  status: ProcessStatus;
}

export interface ProcessProgress {
  progress: number;
  status: ProcessStatus;
  error?: string | null;
}

export type AppStep = 'upload' | 'reference' | 'analyzing' | 'select' | 'processing' | 'complete';
