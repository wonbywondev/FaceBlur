import { Check, User, Eye, EyeOff } from 'lucide-react';
import type { DetectedFace } from '../types';

interface FaceGridProps {
  faces: DetectedFace[];
  selectedFaces: Map<string, boolean>;
  onToggleFace: (faceId: string) => void;
  onBlurAll: () => void;
}

export function FaceGrid({ faces, selectedFaces, onToggleFace, onBlurAll }: FaceGridProps) {
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            감지된 얼굴 ({faces.length}명)
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            블러 처리할 얼굴을 선택하세요. 본인은 자동으로 제외됩니다.
          </p>
        </div>

        <button
          onClick={onBlurAll}
          className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors text-sm font-medium"
        >
          <EyeOff className="w-4 h-4 inline-block mr-2" />
          본인 외 모두 블러
        </button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {faces.map((face) => {
          const isBlurEnabled = selectedFaces.get(face.face_id) ?? face.blur_enabled;

          return (
            <div
              key={face.face_id}
              onClick={() => onToggleFace(face.face_id)}
              className={`
                relative rounded-xl overflow-hidden cursor-pointer
                transition-all duration-200 transform hover:scale-105
                ${face.is_reference ? 'ring-2 ring-green-500' : ''}
                ${isBlurEnabled ? 'ring-2 ring-red-500' : ''}
              `}
            >
              {/* Thumbnail */}
              <div className="aspect-square bg-gray-200 dark:bg-gray-700">
                {face.thumbnail ? (
                  <img
                    src={`data:image/jpeg;base64,${face.thumbnail}`}
                    alt={`Face ${face.face_id}`}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <User className="w-12 h-12 text-gray-400" />
                  </div>
                )}
              </div>

              {/* Overlay info */}
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3">
                <div className="flex items-center justify-between">
                  <span className="text-white text-xs font-medium">
                    {face.similarity_to_reference.toFixed(0)}% 유사
                  </span>

                  <div
                    className={`
                      w-6 h-6 rounded-full flex items-center justify-center
                      ${isBlurEnabled ? 'bg-red-500' : 'bg-gray-500'}
                    `}
                  >
                    {isBlurEnabled ? (
                      <EyeOff className="w-3 h-3 text-white" />
                    ) : (
                      <Eye className="w-3 h-3 text-white" />
                    )}
                  </div>
                </div>

                <p className="text-gray-300 text-xs mt-1">
                  첫 등장: {formatTime(face.first_appearance)}
                </p>
              </div>

              {/* Reference badge */}
              {face.is_reference && (
                <div className="absolute top-2 left-2 px-2 py-1 bg-green-500 text-white text-xs font-medium rounded-full flex items-center gap-1">
                  <Check className="w-3 h-3" />
                  본인
                </div>
              )}

              {/* Blur indicator */}
              {isBlurEnabled && !face.is_reference && (
                <div className="absolute top-2 right-2 px-2 py-1 bg-red-500 text-white text-xs font-medium rounded-full">
                  블러
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
