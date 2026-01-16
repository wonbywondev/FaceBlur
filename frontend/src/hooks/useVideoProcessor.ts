import { create } from 'zustand';
import type {
  AppStep,
  VideoUploadResponse,
  ReferenceUploadResponse,
  AnalysisResult,
  BlurType,
} from '../types';

interface VideoProcessorState {
  // Current step
  step: AppStep;
  setStep: (step: AppStep) => void;

  // Video data
  videoData: VideoUploadResponse | null;
  setVideoData: (data: VideoUploadResponse | null) => void;

  // Reference data
  referenceData: ReferenceUploadResponse | null;
  setReferenceData: (data: ReferenceUploadResponse | null) => void;

  // Analysis
  analysisId: string | null;
  setAnalysisId: (id: string | null) => void;
  analysisProgress: number;
  setAnalysisProgress: (progress: number) => void;
  analysisResult: AnalysisResult | null;
  setAnalysisResult: (result: AnalysisResult | null) => void;

  // Face selection
  selectedFaces: Map<string, boolean>;
  toggleFace: (faceId: string) => void;
  setFaceBlur: (faceId: string, enabled: boolean) => void;
  blurAllExceptReference: () => void;

  // Blur settings
  blurType: BlurType;
  setBlurType: (type: BlurType) => void;
  blurIntensity: number;
  setBlurIntensity: (intensity: number) => void;

  // Processing
  processId: string | null;
  setProcessId: (id: string | null) => void;
  processProgress: number;
  setProcessProgress: (progress: number) => void;

  // Error handling
  error: string | null;
  setError: (error: string | null) => void;

  // Reset
  reset: () => void;
}

const initialState = {
  step: 'upload' as AppStep,
  videoData: null,
  referenceData: null,
  analysisId: null,
  analysisProgress: 0,
  analysisResult: null,
  selectedFaces: new Map<string, boolean>(),
  blurType: 'gaussian' as BlurType,
  blurIntensity: 25,
  processId: null,
  processProgress: 0,
  error: null,
};

export const useVideoProcessor = create<VideoProcessorState>((set, get) => ({
  ...initialState,

  setStep: (step) => set({ step }),

  setVideoData: (videoData) => set({ videoData }),

  setReferenceData: (referenceData) => set({ referenceData }),

  setAnalysisId: (analysisId) => set({ analysisId }),

  setAnalysisProgress: (analysisProgress) => set({ analysisProgress }),

  setAnalysisResult: (analysisResult) => {
    if (analysisResult) {
      // Initialize selectedFaces based on analysis result
      const selectedFaces = new Map<string, boolean>();
      analysisResult.faces.forEach((face) => {
        selectedFaces.set(face.face_id, face.blur_enabled);
      });
      set({ analysisResult, selectedFaces });
    } else {
      set({ analysisResult, selectedFaces: new Map() });
    }
  },

  toggleFace: (faceId) => {
    const { selectedFaces } = get();
    const newSelectedFaces = new Map(selectedFaces);
    newSelectedFaces.set(faceId, !selectedFaces.get(faceId));
    set({ selectedFaces: newSelectedFaces });
  },

  setFaceBlur: (faceId, enabled) => {
    const { selectedFaces } = get();
    const newSelectedFaces = new Map(selectedFaces);
    newSelectedFaces.set(faceId, enabled);
    set({ selectedFaces: newSelectedFaces });
  },

  blurAllExceptReference: () => {
    const { analysisResult, selectedFaces } = get();
    if (!analysisResult) return;

    const newSelectedFaces = new Map(selectedFaces);
    analysisResult.faces.forEach((face) => {
      newSelectedFaces.set(face.face_id, !face.is_reference);
    });
    set({ selectedFaces: newSelectedFaces });
  },

  setBlurType: (blurType) => set({ blurType }),

  setBlurIntensity: (blurIntensity) => set({ blurIntensity }),

  setProcessId: (processId) => set({ processId }),

  setProcessProgress: (processProgress) => set({ processProgress }),

  setError: (error) => set({ error }),

  reset: () => set(initialState),
}));
