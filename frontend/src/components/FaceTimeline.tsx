import { useRef, useEffect } from 'react';
import { User, EyeOff, Eye } from 'lucide-react';
import type { DetectedFace } from '../types';

interface FaceTimelineProps {
  faces: DetectedFace[];
  videoDuration: number;
  selectedFaces: Map<string, boolean>;
  onToggleFace: (faceId: string) => void;
}

interface TimeSlot {
  time: number;
  faces: Array<{
    face: DetectedFace;
    isActive: boolean;
  }>;
}

export function FaceTimeline({ faces, videoDuration, selectedFaces, onToggleFace }: FaceTimelineProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Create time slots (every 1 second)
  const slotInterval = 1; // 1 second intervals
  const totalSlots = Math.ceil(videoDuration / slotInterval);

  // Build time slots with faces active at each time
  const timeSlots: TimeSlot[] = [];
  for (let i = 0; i <= totalSlots; i++) {
    const time = i * slotInterval;
    const activeFaces: TimeSlot['faces'] = [];

    faces.forEach((face) => {
      // Check if face is visible at this time
      const isActive = face.appearances.some(
        (app) => app.start <= time && time <= app.end
      );
      if (isActive) {
        activeFaces.push({ face, isActive: true });
      }
    });

    // Sort by face_id for consistent stacking order
    activeFaces.sort((a, b) => a.face.face_id.localeCompare(b.face.face_id));

    timeSlots.push({ time, faces: activeFaces });
  }

  // Auto-scroll to show content
  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollLeft = 0;
    }
  }, []);

  // Calculate pixel width per slot
  const slotWidth = 48; // pixels per slot
  const totalWidth = totalSlots * slotWidth;

  return (
    <div className="space-y-4">
      {/* Header with legend */}
      <div className="flex items-center justify-between">
        <h4 className="font-medium text-gray-900 dark:text-white">
          타임라인 뷰
        </h4>
        <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-blue-400" />
            <span>표시</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-red-400" />
            <span>블러</span>
          </div>
        </div>
      </div>

      {/* Scrollable timeline container */}
      <div
        ref={scrollContainerRef}
        className="overflow-x-auto pb-2"
        style={{ scrollbarWidth: 'thin' }}
      >
        <div
          className="relative"
          style={{ width: `${Math.max(totalWidth, 800)}px`, minHeight: '280px' }}
        >
          {/* Time axis at TOP */}
          <div className="absolute top-0 left-0 right-0 h-8 border-b border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800/50">
            {/* Time markers every 5 seconds */}
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
                  <div className="w-px h-2 bg-gray-400 dark:bg-gray-500 mt-1" />
                </div>
              )
            )}
          </div>

          {/* Face towers BELOW the time axis */}
          <div className="absolute top-10 left-0 right-0 flex">
            {timeSlots.map((slot, slotIndex) => (
              <div
                key={slotIndex}
                className="flex-shrink-0 flex flex-col items-center gap-1 pt-2"
                style={{ width: `${slotWidth}px` }}
              >
                {slot.faces.map(({ face }) => {
                  const isBlurEnabled = selectedFaces.get(face.face_id) ?? face.blur_enabled;

                  return (
                    <div
                      key={face.face_id}
                      onClick={() => onToggleFace(face.face_id)}
                      className={`
                        relative w-10 h-10 rounded-lg overflow-hidden cursor-pointer
                        transition-all duration-150 hover:scale-110 hover:z-10
                        ${isBlurEnabled
                          ? 'ring-2 ring-red-500 shadow-red-200'
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

                      {/* Small blur indicator */}
                      <div
                        className={`
                          absolute bottom-0 right-0 w-3 h-3 rounded-tl
                          flex items-center justify-center
                          ${isBlurEnabled ? 'bg-red-500' : 'bg-gray-400'}
                        `}
                      >
                        {isBlurEnabled ? (
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

      {/* Summary stats */}
      <div className="flex items-center gap-6 text-sm text-gray-600 dark:text-gray-400 pt-2 border-t border-gray-200 dark:border-gray-700">
        <span>총 {faces.length}명 감지</span>
        <span>영상 길이: {formatTime(videoDuration)}</span>
        <span>
          블러 대상: {Array.from(selectedFaces.values()).filter(Boolean).length}명
        </span>
      </div>

      {/* Scroll hint */}
      {totalWidth > 800 && (
        <p className="text-xs text-gray-400 dark:text-gray-500 text-center">
          ← 좌우로 스크롤하여 전체 타임라인을 확인하세요 →
        </p>
      )}
    </div>
  );
}
