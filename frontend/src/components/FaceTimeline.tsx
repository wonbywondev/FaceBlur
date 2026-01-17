import { useRef, useEffect } from 'react';
import { User, EyeOff, Eye, Star, UserCheck } from 'lucide-react';
import type { DetectedFace } from '../types';

interface FaceTimelineProps {
  faces: DetectedFace[];
  videoDuration: number;
  selectedFaces: Map<string, boolean>;
  myFaceId: string | null;
  onToggleFace: (faceId: string) => void;
  onBlurAll: () => void;
  onSetMyFace: (faceId: string | null) => void;
}

interface TimeSlot {
  time: number;
  faces: Array<{
    face: DetectedFace;
    isActive: boolean;
  }>;
}

export function FaceTimeline({
  faces,
  videoDuration,
  selectedFaces,
  myFaceId,
  onToggleFace,
  onBlurAll,
  onSetMyFace,
}: FaceTimelineProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);

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

  // Create time slots (every 1 second)
  const slotInterval = 1;
  const totalSlots = Math.ceil(videoDuration / slotInterval);

  // Build time slots with faces active at each time
  const timeSlots: TimeSlot[] = [];
  for (let i = 0; i <= totalSlots; i++) {
    const time = i * slotInterval;
    const activeFaces: TimeSlot['faces'] = [];

    faces.forEach((face) => {
      const isActive = face.appearances.some(
        (app) => app.start <= time && time <= app.end
      );
      if (isActive) {
        activeFaces.push({ face, isActive: true });
      }
    });

    activeFaces.sort((a, b) => a.face.face_id.localeCompare(b.face.face_id));
    timeSlots.push({ time, faces: activeFaces });
  }

  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollLeft = 0;
    }
  }, []);

  const slotWidth = 48;
  const totalWidth = totalSlots * slotWidth;

  const handleMyFaceClick = (e: React.MouseEvent, faceId: string) => {
    e.stopPropagation();
    if (myFaceId === faceId) {
      onSetMyFace(null);
    } else {
      onSetMyFace(faceId);
    }
  };

  return (
    <div className="space-y-4">
      {/* Header with controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            타임라인 뷰 ({faces.length}명)
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            시간대별 등장 얼굴 확인 및 블러 설정
          </p>
        </div>
        <button
          onClick={onBlurAll}
          className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors text-sm font-medium"
        >
          <EyeOff className="w-4 h-4 inline-block mr-2" />
          모두 블러
        </button>
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
          <div className="grid grid-cols-5 gap-3">
            {topFaces.map((face) => {
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
                    ${!isMyFace && !isBlurEnabled ? 'ring-2 ring-yellow-400' : ''}
                  `}
                >
                  <div className="aspect-square bg-gray-200 dark:bg-gray-700 min-h-[80px]">
                    {face.thumbnail ? (
                      <img
                        src={`data:image/jpeg;base64,${face.thumbnail}`}
                        alt={`Face ${face.face_id}`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <User className="w-8 h-8 text-gray-400" />
                      </div>
                    )}
                  </div>

                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                    <div className="flex items-center justify-between">
                      <span className="text-white text-xs font-medium">
                        {appearanceCount}회
                      </span>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={(e) => handleMyFaceClick(e, face.face_id)}
                          className={`
                            w-5 h-5 rounded-full flex items-center justify-center
                            transition-colors
                            ${isMyFace ? 'bg-green-500' : 'bg-gray-600 hover:bg-gray-500'}
                          `}
                          title={isMyFace ? '내 얼굴 해제' : '내 얼굴로 지정'}
                        >
                          <UserCheck className="w-3 h-3 text-white" />
                        </button>
                        <div
                          className={`
                            w-5 h-5 rounded-full flex items-center justify-center
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
                  </div>

                  {isMyFace && (
                    <div className="absolute top-1 left-1 px-1.5 py-0.5 bg-green-500 text-white text-[10px] font-medium rounded-full flex items-center gap-0.5">
                      <UserCheck className="w-2.5 h-2.5" />
                      내 얼굴
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Timeline section */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h4 className="font-medium text-gray-700 dark:text-gray-300">
            시간대별 등장
          </h4>
          <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded ring-1 ring-gray-300" />
              <span>표시</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded ring-2 ring-red-500" />
              <span>블러</span>
            </div>
          </div>
        </div>

        {/* Scrollable timeline container */}
        <div
          ref={scrollContainerRef}
          className="overflow-x-auto pb-2 border border-gray-200 dark:border-gray-700 rounded-xl"
          style={{ scrollbarWidth: 'thin' }}
        >
          <div
            className="relative"
            style={{ width: `${Math.max(totalWidth, 800)}px`, minHeight: '200px' }}
          >
            {/* Time axis at TOP */}
            <div className="sticky top-0 left-0 right-0 h-7 border-b border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800/80 z-10">
              {Array.from({ length: Math.ceil(videoDuration / 5) + 1 }, (_, i) => i * 5).map(
                (time) => (
                  <div
                    key={time}
                    className="absolute top-0 transform -translate-x-1/2"
                    style={{ left: `${(time / videoDuration) * 100}%` }}
                  >
                    <span className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap block pt-1">
                      {formatTime(time)}
                    </span>
                    <div className="w-px h-2 bg-gray-400 dark:bg-gray-500 mt-0.5" />
                  </div>
                )
              )}
            </div>

            {/* Face towers BELOW the time axis */}
            <div className="absolute top-8 left-0 right-0 flex">
              {timeSlots.map((slot, slotIndex) => (
                <div
                  key={slotIndex}
                  className="flex-shrink-0 flex flex-col items-center gap-1 pt-1"
                  style={{ width: `${slotWidth}px` }}
                >
                  {slot.faces.map(({ face }) => {
                    const isBlurEnabled = selectedFaces.get(face.face_id) ?? face.blur_enabled;
                    const isMyFace = myFaceId === face.face_id;

                    return (
                      <div
                        key={face.face_id}
                        onClick={() => onToggleFace(face.face_id)}
                        className={`
                          relative w-9 h-9 rounded-lg overflow-hidden cursor-pointer
                          transition-all duration-150 hover:scale-110 hover:z-10
                          ${isMyFace
                            ? 'ring-2 ring-green-500'
                            : isBlurEnabled
                              ? 'ring-2 ring-red-500'
                              : 'ring-1 ring-gray-300 dark:ring-gray-600'
                          }
                        `}
                        title={`${face.face_id} - ${formatTime(slot.time)}`}
                      >
                        {face.thumbnail ? (
                          <img
                            src={`data:image/jpeg;base64,${face.thumbnail}`}
                            alt={face.face_id}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center bg-gray-200 dark:bg-gray-700">
                            <User className="w-4 h-4 text-gray-400" />
                          </div>
                        )}

                        <div
                          className={`
                            absolute bottom-0 right-0 w-3 h-3 rounded-tl
                            flex items-center justify-center
                            ${isMyFace ? 'bg-green-500' : isBlurEnabled ? 'bg-red-500' : 'bg-gray-400'}
                          `}
                        >
                          {isMyFace ? (
                            <UserCheck className="w-2 h-2 text-white" />
                          ) : isBlurEnabled ? (
                            <EyeOff className="w-2 h-2 text-white" />
                          ) : (
                            <Eye className="w-2 h-2 text-white" />
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Summary stats */}
      <div className="flex items-center gap-6 text-sm text-gray-600 dark:text-gray-400 pt-2 border-t border-gray-200 dark:border-gray-700">
        <span>총 {faces.length}명 감지</span>
        <span>영상 길이: {formatTime(videoDuration)}</span>
        <span>
          블러 대상: {Array.from(selectedFaces.values()).filter(Boolean).length}명
        </span>
      </div>

      {totalWidth > 800 && (
        <p className="text-xs text-gray-400 dark:text-gray-500 text-center">
          ← 좌우로 스크롤하여 전체 타임라인을 확인하세요 →
        </p>
      )}
    </div>
  );
}
