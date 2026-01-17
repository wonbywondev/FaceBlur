import { Check, User, Eye, EyeOff, Star, UserCheck } from 'lucide-react';
import type { DetectedFace } from '../types';

interface FaceGridProps {
  faces: DetectedFace[];
  selectedFaces: Map<string, boolean>;
  myFaceId: string | null;
  onToggleFace: (faceId: string) => void;
  onBlurAll: () => void;
  onSetMyFace: (faceId: string | null) => void;
}

export function FaceGrid({
  faces,
  selectedFaces,
  myFaceId,
  onToggleFace,
  onBlurAll,
  onSetMyFace,
}: FaceGridProps) {
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Sort faces by appearance count and get top 5
  const sortedFaces = [...faces].sort((a, b) =>
    (b.appearance_count || b.appearances.length) - (a.appearance_count || a.appearances.length)
  );
  const topFaces = sortedFaces.slice(0, 5);
  const otherFaces = sortedFaces.slice(5);

  const handleMyFaceClick = (e: React.MouseEvent, faceId: string) => {
    e.stopPropagation();
    if (myFaceId === faceId) {
      onSetMyFace(null);
    } else {
      onSetMyFace(faceId);
    }
  };

  const renderFaceCard = (face: DetectedFace, isTop: boolean = false) => {
    const isBlurEnabled = selectedFaces.get(face.face_id) ?? face.blur_enabled;
    const isMyFace = myFaceId === face.face_id;
    const appearanceCount = face.appearance_count || face.appearances.length;

    return (
      <div
        key={face.face_id}
        onClick={() => onToggleFace(face.face_id)}
        className={`
          relative rounded-xl overflow-hidden cursor-pointer
          transition-all duration-200 transform hover:scale-105
          ${isMyFace ? 'ring-2 ring-green-500 ring-offset-2' : ''}
          ${isBlurEnabled && !isMyFace ? 'ring-2 ring-red-500' : ''}
          ${isTop && !isMyFace && !isBlurEnabled ? 'ring-2 ring-yellow-400' : ''}
        `}
      >
        {/* Thumbnail */}
        <div className={`aspect-square bg-gray-200 dark:bg-gray-700 ${isTop ? 'min-h-[120px]' : ''}`}>
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
              {appearanceCount}회 등장
            </span>

            <div className="flex items-center gap-1">
              {/* My face toggle */}
              <button
                onClick={(e) => handleMyFaceClick(e, face.face_id)}
                className={`
                  w-6 h-6 rounded-full flex items-center justify-center
                  transition-colors
                  ${isMyFace ? 'bg-green-500' : 'bg-gray-600 hover:bg-gray-500'}
                `}
                title={isMyFace ? '내 얼굴 해제' : '내 얼굴로 지정'}
              >
                <UserCheck className="w-3 h-3 text-white" />
              </button>

              {/* Blur toggle indicator */}
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
          </div>

          <p className="text-gray-300 text-xs mt-1">
            첫 등장: {formatTime(face.first_appearance)}
          </p>
        </div>

        {/* My Face badge */}
        {isMyFace && (
          <div className="absolute top-2 left-2 px-2 py-1 bg-green-500 text-white text-xs font-medium rounded-full flex items-center gap-1">
            <UserCheck className="w-3 h-3" />
            내 얼굴
          </div>
        )}

        {/* Top face badge */}
        {isTop && !isMyFace && (
          <div className="absolute top-2 left-2 px-2 py-1 bg-yellow-500 text-white text-xs font-medium rounded-full flex items-center gap-1">
            <Star className="w-3 h-3" />
            주요
          </div>
        )}

        {/* Reference badge */}
        {face.is_reference && !isMyFace && (
          <div className="absolute top-2 left-2 px-2 py-1 bg-green-500 text-white text-xs font-medium rounded-full flex items-center gap-1">
            <Check className="w-3 h-3" />
            본인
          </div>
        )}

        {/* Blur indicator */}
        {isBlurEnabled && !isMyFace && (
          <div className="absolute top-2 right-2 px-2 py-1 bg-red-500 text-white text-xs font-medium rounded-full">
            블러
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            감지된 얼굴 ({faces.length}명)
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            블러 처리할 얼굴을 선택하세요. <UserCheck className="w-3 h-3 inline" /> 버튼으로 내 얼굴을 지정할 수 있습니다.
          </p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={onBlurAll}
            className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors text-sm font-medium"
          >
            <EyeOff className="w-4 h-4 inline-block mr-2" />
            모두 블러
          </button>
        </div>
      </div>

      {/* My face info */}
      {myFaceId && (
        <div className="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl">
          <p className="text-sm text-green-700 dark:text-green-300 flex items-center gap-2">
            <UserCheck className="w-4 h-4" />
            내 얼굴이 지정되었습니다. "모두 블러" 클릭 시 내 얼굴을 제외한 모든 얼굴이 블러 처리됩니다.
          </p>
        </div>
      )}

      {/* Top 5 Faces Section */}
      {topFaces.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Star className="w-5 h-5 text-yellow-500" />
            <h4 className="font-medium text-gray-900 dark:text-white">
              자주 등장하는 얼굴 (Top {topFaces.length})
            </h4>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
            {topFaces.map((face) => renderFaceCard(face, true))}
          </div>
        </div>
      )}

      {/* Other Faces Section */}
      {otherFaces.length > 0 && (
        <div className="space-y-3">
          <h4 className="font-medium text-gray-700 dark:text-gray-300">
            기타 얼굴 ({otherFaces.length}명)
          </h4>
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
            {otherFaces.map((face) => renderFaceCard(face, false))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {faces.length === 0 && (
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          감지된 얼굴이 없습니다
        </div>
      )}
    </div>
  );
}
