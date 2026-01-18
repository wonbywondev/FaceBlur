import axios from 'axios';
import type {
  VideoUploadResponse,
  ReferenceUploadResponse,
  AnalysisResult,
  BlurSettings,
  AnalysisProgress,
  ProcessProgress,
  ExpectedPersonCount,
} from '../types';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 300000, // 5 minutes for long operations
});

export const uploadVideo = async (file: File): Promise<VideoUploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<VideoUploadResponse>('/upload/video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const uploadReference = async (
  videoId: string,
  files: File[]
): Promise<ReferenceUploadResponse> => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  const response = await api.post<ReferenceUploadResponse>(
    `/upload/reference?video_id=${encodeURIComponent(videoId)}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};

export const startAnalysis = async (
  videoId: string,
  referenceId?: string,
  expectedPersons: ExpectedPersonCount = '10'
): Promise<{ analysis_id: string; estimated_time: number }> => {
  const payload: { video_id: string; reference_id?: string; expected_persons: string } = {
    video_id: videoId,
    expected_persons: expectedPersons,
  };
  if (referenceId) {
    payload.reference_id = referenceId;
  }

  const response = await api.post('/analyze', payload);
  return response.data;
};

export const getAnalysisStatus = async (analysisId: string): Promise<AnalysisProgress> => {
  const response = await api.get<AnalysisProgress>(`/analyze/${analysisId}/status`);
  return response.data;
};

export const getAnalysisResult = async (analysisId: string): Promise<AnalysisResult> => {
  const response = await api.get<AnalysisResult>(`/analyze/${analysisId}/result`);
  return response.data;
};

export const startBlurProcessing = async (
  analysisId: string,
  blurSettings: BlurSettings
): Promise<{ process_id: string }> => {
  const response = await api.post('/process/blur', {
    analysis_id: analysisId,
    blur_settings: blurSettings,
  });

  return response.data;
};

export const getProcessStatus = async (processId: string): Promise<ProcessProgress> => {
  const response = await api.get<ProcessProgress>(`/process/${processId}/status`);
  return response.data;
};

export const getDownloadUrl = (processId: string): string => {
  return `/api/v1/process/${processId}/download`;
};

export const stopProcessing = async (processId: string): Promise<{ status: string }> => {
  const response = await api.post(`/process/${processId}/stop`);
  return response.data;
};

export const pauseProcessing = async (processId: string): Promise<{ status: string }> => {
  const response = await api.post(`/process/${processId}/pause`);
  return response.data;
};

export const resumeProcessing = async (processId: string): Promise<{ status: string }> => {
  const response = await api.post(`/process/${processId}/resume`);
  return response.data;
};

export const getServiceInfo = async () => {
  const response = await api.get('/info');
  return response.data;
};
