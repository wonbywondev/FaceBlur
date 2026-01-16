import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image, Video, AlertCircle } from 'lucide-react';

interface UploadZoneProps {
  type: 'video' | 'reference';
  onUpload: (files: File[]) => void;
  isUploading?: boolean;
  error?: string | null;
}

export function UploadZone({ type, onUpload, isUploading = false, error }: UploadZoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onUpload(acceptedFiles);
      }
    },
    [onUpload]
  );

  const accept: Record<string, string[]> =
    type === 'video'
      ? { 'video/mp4': ['.mp4'], 'video/quicktime': ['.mov'] }
      : { 'image/jpeg': ['.jpg', '.jpeg'], 'image/png': ['.png'] };

  const maxFiles = type === 'video' ? 1 : 5;
  const maxSize = type === 'video' ? 500 * 1024 * 1024 : 10 * 1024 * 1024;

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept,
    maxFiles,
    maxSize,
    disabled: isUploading,
  });

  const Icon = type === 'video' ? Video : Image;

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-xl p-8 md:p-12 text-center cursor-pointer
          transition-all duration-200 ease-in-out
          ${isDragActive && !isDragReject ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20' : ''}
          ${isDragReject ? 'border-red-500 bg-red-50 dark:bg-red-900/20' : ''}
          ${!isDragActive && !isDragReject ? 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500 bg-white dark:bg-gray-800' : ''}
          ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center gap-4">
          <div
            className={`
              w-16 h-16 rounded-full flex items-center justify-center
              ${isDragActive ? 'bg-primary-100 dark:bg-primary-800' : 'bg-gray-100 dark:bg-gray-700'}
            `}
          >
            <Icon
              className={`w-8 h-8 ${isDragActive ? 'text-primary-600' : 'text-gray-500 dark:text-gray-400'}`}
            />
          </div>

          {isUploading ? (
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-gray-600 dark:text-gray-300">업로드 중...</span>
            </div>
          ) : isDragActive ? (
            <p className="text-primary-600 dark:text-primary-400 font-medium">파일을 놓으세요</p>
          ) : (
            <>
              <div>
                <p className="text-gray-700 dark:text-gray-200 font-medium">
                  {type === 'video' ? '영상 파일을 드래그하거나 클릭하세요' : '본인 사진을 드래그하거나 클릭하세요'}
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  {type === 'video' ? 'MP4/MOV, 최대 500MB, 10분 이하' : 'JPG/PNG, 1-5장'}
                </p>
              </div>

              <button
                type="button"
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
              >
                <Upload className="w-4 h-4 inline-block mr-2" />
                파일 선택
              </button>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-3 flex items-center gap-2 text-red-600 dark:text-red-400">
          <AlertCircle className="w-4 h-4" />
          <span className="text-sm">{error}</span>
        </div>
      )}
    </div>
  );
}
