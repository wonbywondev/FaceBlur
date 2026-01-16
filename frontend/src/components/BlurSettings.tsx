import { Circle, Grid3X3, Square } from 'lucide-react';
import type { BlurType } from '../types';

interface BlurSettingsProps {
  blurType: BlurType;
  intensity: number;
  onTypeChange: (type: BlurType) => void;
  onIntensityChange: (intensity: number) => void;
}

export function BlurSettings({
  blurType,
  intensity,
  onTypeChange,
  onIntensityChange,
}: BlurSettingsProps) {
  const blurTypes: { type: BlurType; label: string; icon: typeof Circle; description: string }[] = [
    {
      type: 'gaussian',
      label: 'Gaussian',
      icon: Circle,
      description: '부드러운 블러',
    },
    {
      type: 'mosaic',
      label: 'Mosaic',
      icon: Grid3X3,
      description: '픽셀화 효과',
    },
    {
      type: 'blackout',
      label: 'Blackout',
      icon: Square,
      description: '완전 가림',
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">블러 스타일</h4>
        <div className="grid grid-cols-3 gap-3">
          {blurTypes.map(({ type, label, icon: Icon, description }) => (
            <button
              key={type}
              onClick={() => onTypeChange(type)}
              className={`
                p-4 rounded-xl border-2 transition-all duration-200
                ${
                  blurType === type
                    ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }
              `}
            >
              <Icon
                className={`w-6 h-6 mx-auto mb-2 ${
                  blurType === type ? 'text-primary-600' : 'text-gray-500 dark:text-gray-400'
                }`}
              />
              <p
                className={`text-sm font-medium ${
                  blurType === type ? 'text-primary-700 dark:text-primary-300' : 'text-gray-700 dark:text-gray-300'
                }`}
              >
                {label}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{description}</p>
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="flex justify-between items-center mb-3">
          <h4 className="text-sm font-medium text-gray-900 dark:text-white">블러 강도</h4>
          <span className="text-sm text-gray-500 dark:text-gray-400">{intensity}</span>
        </div>
        <input
          type="range"
          min="1"
          max="50"
          value={intensity}
          onChange={(e) => onIntensityChange(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
        />
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>약함</span>
          <span>강함</span>
        </div>
      </div>
    </div>
  );
}
