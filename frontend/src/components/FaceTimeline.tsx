import { User, EyeOff, Eye } from 'lucide-react';
import type { DetectedFace } from '../types';

interface FaceTimelineProps {
  faces: DetectedFace[];
  videoDuration: number;
  selectedFaces: Map<string, boolean>;
  onToggleFace: (faceId: string) => void;
}

export function FaceTimeline({ faces, videoDuration, selectedFaces, onToggleFace }: FaceTimelineProps) {
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Sort faces by first appearance
  const sortedFaces = [...faces].sort((a, b) => a.first_appearance - b.first_appearance);

  // Generate time markers
  const markerCount = Math.min(10, Math.ceil(videoDuration / 10));
  const markers = Array.from({ length: markerCount + 1 }, (_, i) =>
    (videoDuration / markerCount) * i
  );

  return (
    <div className="space-y-4">
      {/* Time axis */}
      <div className="relative h-6 ml-20">
        <div className="absolute inset-x-0 top-3 h-px bg-gray-300 dark:bg-gray-600" />
        {markers.map((time, idx) => (
          <div
            key={idx}
            className="absolute transform -translate-x-1/2"
            style={{ left: `${(time / videoDuration) * 100}%` }}
          >
            <div className="w-px h-2 bg-gray-400 dark:bg-gray-500" />
            <span className="text-xs text-gray-500 dark:text-gray-400 mt-1 block whitespace-nowrap">
              {formatTime(time)}
            </span>
          </div>
        ))}
      </div>

      {/* Face rows */}
      <div className="space-y-2">
        {sortedFaces.map((face) => {
          const isBlurEnabled = selectedFaces.get(face.face_id) ?? face.blur_enabled;

          return (
            <div key={face.face_id} className="flex items-center gap-3">
              {/* Face thumbnail */}
              <div
                onClick={() => onToggleFace(face.face_id)}
                className={`
                  w-14 h-14 rounded-lg overflow-hidden cursor-pointer flex-shrink-0
                  transition-all duration-200 hover:scale-105
                  ${isBlurEnabled ? 'ring-2 ring-red-500' : 'ring-1 ring-gray-300 dark:ring-gray-600'}
                `}
              >
                {face.thumbnail ? (
                  <img
                    src={`data:image/jpeg;base64,${face.thumbnail}`}
                    alt={`Face ${face.face_id}`}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-gray-200 dark:bg-gray-700">
                    <User className="w-6 h-6 text-gray-400" />
                  </div>
                )}
              </div>

              {/* Timeline bar */}
              <div className="flex-1 relative h-8 bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden">
                {face.appearances.map((app, idx) => {
                  const startPercent = (app.start / videoDuration) * 100;
                  const widthPercent = Math.max(0.5, ((app.end - app.start) / videoDuration) * 100);

                  return (
                    <div
                      key={idx}
                      className={`
                        absolute top-1 bottom-1 rounded
                        ${isBlurEnabled ? 'bg-red-400' : 'bg-blue-400'}
                        opacity-80 hover:opacity-100 transition-opacity
                      `}
                      style={{
                        left: `${startPercent}%`,
                        width: `${widthPercent}%`,
                        minWidth: '4px',
                      }}
                      title={`${formatTime(app.start)} - ${formatTime(app.end)}`}
                    />
                  );
                })}
              </div>

              {/* Blur toggle */}
              <button
                onClick={() => onToggleFace(face.face_id)}
                className={`
                  w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0
                  transition-colors
                  ${isBlurEnabled
                    ? 'bg-red-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-600 text-gray-600 dark:text-gray-300'
                  }
                `}
              >
                {isBlurEnabled ? (
                  <EyeOff className="w-4 h-4" />
                ) : (
                  <Eye className="w-4 h-4" />
                )}
              </button>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400 pt-2 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-blue-400" />
          <span>표시</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-red-400" />
          <span>블러 처리</span>
        </div>
      </div>
    </div>
  );
}
