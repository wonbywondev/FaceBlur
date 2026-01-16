import { useState, useEffect } from 'react';
import {
  Video,
  Scan,
  Check,
  Download,
  ArrowRight,
  Loader2,
  AlertCircle,
  RefreshCw,
  Play,
  X,
} from 'lucide-react';

import { UploadZone } from './components/UploadZone';
import { FaceGrid } from './components/FaceGrid';
import { ProgressBar } from './components/ProgressBar';
import { BlurSettings } from './components/BlurSettings';
import { useVideoProcessor } from './hooks/useVideoProcessor';
import * as api from './services/api';

function App() {
  const {
    step,
    setStep,
    videoData,
    setVideoData,
    analysisId,
    setAnalysisId,
    analysisProgress,
    setAnalysisProgress,
    analysisResult,
    setAnalysisResult,
    selectedFaces,
    toggleFace,
    blurAllExceptReference,
    blurType,
    setBlurType,
    blurIntensity,
    setBlurIntensity,
    processId,
    setProcessId,
    processProgress,
    setProcessProgress,
    error,
    setError,
    reset,
  } = useVideoProcessor();

  const [isUploading, setIsUploading] = useState(false);

  // Poll for analysis progress
  useEffect(() => {
    if (step !== 'analyzing' || !analysisId) return;

    const interval = setInterval(async () => {
      try {
        const status = await api.getAnalysisStatus(analysisId);
        setAnalysisProgress(status.progress);

        if (status.status === 'completed') {
          const result = await api.getAnalysisResult(analysisId);
          setAnalysisResult(result);
          setStep('select');
        } else if (status.status === 'failed') {
          setError('분석 중 오류가 발생했습니다.');
          setStep('upload');
        }
      } catch (err) {
        console.error('Error polling analysis status:', err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [step, analysisId]);

  // Poll for processing progress
  useEffect(() => {
    console.log('[POLLING] useEffect triggered, step:', step, 'processId:', processId);

    if (step !== 'processing' || !processId) {
      console.log('[POLLING] Skipping - step is not processing or no processId');
      return;
    }

    console.log('[POLLING] Starting polling for processId:', processId);

    const interval = setInterval(async () => {
      try {
        console.log('[POLLING] Fetching status...');
        const status = await api.getProcessStatus(processId);
        console.log('[POLLING] Status received:', status);
        setProcessProgress(status.progress);

        if (status.status === 'completed') {
          console.log('[POLLING] Processing completed!');
          setStep('complete');
        } else if (status.status === 'failed') {
          console.log('[POLLING] Processing failed:', status.error);
          setError(status.error || '처리 중 오류가 발생했습니다.');
          setStep('select');
        }
      } catch (err) {
        console.error('[POLLING] Error polling process status:', err);
      }
    }, 1000);

    return () => {
      console.log('[POLLING] Cleaning up interval');
      clearInterval(interval);
    };
  }, [step, processId, setProcessProgress, setStep, setError]);

  const handleVideoUpload = async (files: File[]) => {
    if (files.length === 0) return;

    setIsUploading(true);
    setError(null);

    try {
      const result = await api.uploadVideo(files[0]);
      setVideoData(result);
      setStep('confirm');  // Go to confirmation step
    } catch (err: any) {
      setError(err.response?.data?.detail || '영상 업로드에 실패했습니다.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleConfirmAnalysis = async () => {
    if (!videoData) return;

    setError(null);
    setStep('analyzing');

    try {
      // Start analysis without reference image
      const result = await api.startAnalysis(videoData.video_id);
      setAnalysisId(result.analysis_id);
    } catch (err: any) {
      setError(err.response?.data?.detail || '분석 시작에 실패했습니다.');
      setStep('upload');
    }
  };

  const handleCancelConfirm = () => {
    reset();
  };

  const handleStartProcessing = async () => {
    console.log('[PROCESS] Starting blur processing...');
    if (!analysisId) {
      console.log('[PROCESS] No analysisId, returning');
      return;
    }

    // Get face IDs to blur
    const faceIdsToBlur = Array.from(selectedFaces.entries())
      .filter(([_, enabled]) => enabled)
      .map(([faceId]) => faceId);

    console.log('[PROCESS] Face IDs to blur:', faceIdsToBlur.length);

    if (faceIdsToBlur.length === 0) {
      setError('블러 처리할 얼굴을 선택해주세요.');
      return;
    }

    setError(null);
    setStep('processing');
    console.log('[PROCESS] Step set to processing, calling API...');

    try {
      const result = await api.startBlurProcessing(analysisId, {
        type: blurType,
        intensity: blurIntensity,
        face_ids: faceIdsToBlur,
      });
      console.log('[PROCESS] API returned, process_id:', result.process_id);
      setProcessId(result.process_id);
      console.log('[PROCESS] processId set to:', result.process_id);
    } catch (err: any) {
      console.error('[PROCESS] API error:', err);
      console.error('[PROCESS] Error response:', err.response);
      console.error('[PROCESS] Error status:', err.response?.status);
      console.error('[PROCESS] Error data:', err.response?.data);
      console.error('[PROCESS] Error message:', err.message);
      setError(err.response?.data?.detail || err.message || '처리 시작에 실패했습니다.');
      setStep('select');
    }
  };

  const handleDownload = () => {
    if (!processId) return;
    window.open(api.getDownloadUrl(processId), '_blank');
  };

  const steps = [
    { id: 'upload', label: '영상 업로드', icon: Video },
    { id: 'confirm', label: '확인', icon: Play },
    { id: 'analyzing', label: '분석 중', icon: Scan },
    { id: 'select', label: '얼굴 선택', icon: Check },
    { id: 'processing', label: '처리 중', icon: Loader2 },
    { id: 'complete', label: '완료', icon: Download },
  ];

  const currentStepIndex = steps.findIndex((s) => s.id === step);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-5xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-primary-500 rounded-xl flex items-center justify-center">
                <Video className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900 dark:text-white">Face Blur</h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">AI 얼굴 블러 서비스</p>
              </div>
            </div>

            {step !== 'upload' && (
              <button
                onClick={reset}
                className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
              >
                <RefreshCw className="w-4 h-4" />
                새로 시작
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Progress Steps */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-5xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            {steps.map((s, index) => {
              const Icon = s.icon;
              const isActive = s.id === step;
              const isCompleted = index < currentStepIndex;

              return (
                <div key={s.id} className="flex items-center">
                  <div className="flex flex-col items-center">
                    <div
                      className={`
                        w-10 h-10 rounded-full flex items-center justify-center transition-colors
                        ${isActive ? 'bg-primary-500 text-white' : ''}
                        ${isCompleted ? 'bg-green-500 text-white' : ''}
                        ${!isActive && !isCompleted ? 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400' : ''}
                      `}
                    >
                      {isCompleted ? (
                        <Check className="w-5 h-5" />
                      ) : (
                        <Icon className={`w-5 h-5 ${isActive && s.id === 'processing' ? 'animate-spin' : ''}`} />
                      )}
                    </div>
                    <span
                      className={`text-xs mt-2 ${
                        isActive || isCompleted ? 'text-gray-900 dark:text-white' : 'text-gray-400'
                      }`}
                    >
                      {s.label}
                    </span>
                  </div>

                  {index < steps.length - 1 && (
                    <div
                      className={`w-12 md:w-24 h-0.5 mx-2 ${
                        index < currentStepIndex ? 'bg-green-500' : 'bg-gray-200 dark:bg-gray-700'
                      }`}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <span className="text-red-700 dark:text-red-300">{error}</span>
          </div>
        )}

        {/* Step: Upload */}
        {step === 'upload' && (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">영상 업로드</h2>
              <p className="text-gray-500 dark:text-gray-400 mt-2">
                블러 처리할 영상을 업로드하세요
              </p>
            </div>
            <UploadZone type="video" onUpload={handleVideoUpload} isUploading={isUploading} error={null} />
          </div>
        )}

        {/* Step: Confirm */}
        {step === 'confirm' && videoData && (
          <div className="space-y-6">
            <div className="max-w-lg mx-auto">
              <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden">
                {/* Header */}
                <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between">
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                      이 영상으로 진행할까요?
                    </h2>
                    <button
                      onClick={handleCancelConfirm}
                      className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                {/* Video Info */}
                <div className="p-6">
                  <div className="flex items-start gap-4">
                    <div className="w-16 h-16 bg-primary-100 dark:bg-primary-900/30 rounded-xl flex items-center justify-center flex-shrink-0">
                      <Video className="w-8 h-8 text-primary-500" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-gray-900 dark:text-white truncate">
                        {videoData.filename}
                      </p>
                      <div className="mt-2 flex flex-wrap gap-3 text-sm text-gray-500 dark:text-gray-400">
                        <span>⏱️ {Math.round(videoData.duration)}초</span>
                        <span>📐 {videoData.resolution.width}x{videoData.resolution.height}</span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl">
                    <p className="text-sm text-blue-700 dark:text-blue-300">
                      💡 AI가 영상을 분석하여 등장하는 모든 얼굴을 자동으로 감지합니다.
                      분석 완료 후 블러 처리할 얼굴을 선택할 수 있습니다.
                    </p>
                  </div>
                </div>

                {/* Actions */}
                <div className="p-6 bg-gray-50 dark:bg-gray-900/50 flex gap-3">
                  <button
                    onClick={handleCancelConfirm}
                    className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-xl font-medium hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  >
                    취소
                  </button>
                  <button
                    onClick={handleConfirmAnalysis}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium transition-colors"
                  >
                    분석 시작
                    <ArrowRight className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step: Analyzing */}
        {step === 'analyzing' && (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">영상 분석 중</h2>
              <p className="text-gray-500 dark:text-gray-400 mt-2">
                AI가 영상에서 얼굴을 탐지하고 분석하고 있습니다
              </p>
            </div>

            <div className="max-w-md mx-auto">
              <div className="p-8 bg-white dark:bg-gray-800 rounded-2xl shadow-lg">
                <div className="flex justify-center mb-6">
                  <div className="w-16 h-16 bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center">
                    <Scan className="w-8 h-8 text-primary-500 animate-pulse" />
                  </div>
                </div>

                <ProgressBar progress={analysisProgress} label="분석 진행률" color="primary" />

                <p className="text-center text-sm text-gray-500 dark:text-gray-400 mt-4">
                  잠시만 기다려주세요...
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Step: Select Faces */}
        {step === 'select' && analysisResult && (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">얼굴 선택</h2>
              <p className="text-gray-500 dark:text-gray-400 mt-2">
                블러 처리할 얼굴을 선택하세요
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
              <FaceGrid
                faces={analysisResult.faces}
                selectedFaces={selectedFaces}
                onToggleFace={toggleFace}
                onBlurAll={blurAllExceptReference}
              />
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">블러 설정</h3>
              <BlurSettings
                blurType={blurType}
                intensity={blurIntensity}
                onTypeChange={setBlurType}
                onIntensityChange={setBlurIntensity}
              />
            </div>

            <div className="flex justify-end">
              <button
                onClick={handleStartProcessing}
                className="flex items-center gap-2 px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium transition-colors"
              >
                블러 처리 시작
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}

        {/* Step: Processing */}
        {step === 'processing' && (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">블러 처리 중</h2>
              <p className="text-gray-500 dark:text-gray-400 mt-2">
                선택한 얼굴에 블러를 적용하고 있습니다
              </p>
            </div>

            <div className="max-w-md mx-auto">
              <div className="p-8 bg-white dark:bg-gray-800 rounded-2xl shadow-lg">
                <div className="flex justify-center mb-6">
                  <div className="w-16 h-16 bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center">
                    <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
                  </div>
                </div>

                <ProgressBar progress={processProgress} label="처리 진행률" color="primary" />

                <p className="text-center text-sm text-gray-500 dark:text-gray-400 mt-4">
                  영상을 렌더링하고 있습니다...
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Step: Complete */}
        {step === 'complete' && (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">처리 완료!</h2>
              <p className="text-gray-500 dark:text-gray-400 mt-2">
                블러 처리가 완료되었습니다
              </p>
            </div>

            <div className="max-w-md mx-auto">
              <div className="p-8 bg-white dark:bg-gray-800 rounded-2xl shadow-lg text-center">
                <div className="flex justify-center mb-6">
                  <div className="w-20 h-20 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center">
                    <Check className="w-10 h-10 text-green-500" />
                  </div>
                </div>

                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  영상이 준비되었습니다
                </h3>
                <p className="text-gray-500 dark:text-gray-400 mb-6">
                  아래 버튼을 클릭해서 다운로드하세요
                </p>

                <button
                  onClick={handleDownload}
                  className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-green-500 hover:bg-green-600 text-white rounded-xl font-medium transition-colors"
                >
                  <Download className="w-5 h-5" />
                  영상 다운로드
                </button>

                <button
                  onClick={reset}
                  className="w-full mt-4 flex items-center justify-center gap-2 px-6 py-3 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-xl font-medium hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <RefreshCw className="w-5 h-5" />
                  새 영상 처리하기
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 dark:border-gray-700 mt-auto">
        <div className="max-w-5xl mx-auto px-4 py-6">
          <p className="text-center text-sm text-gray-500 dark:text-gray-400">
            업로드된 영상과 이미지는 24시간 내에 자동 삭제됩니다.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
