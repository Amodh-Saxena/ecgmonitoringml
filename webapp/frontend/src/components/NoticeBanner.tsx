import React from "react";
import { AlertCircle, AlertTriangle, Info, X } from "lucide-react";

interface NoticeBannerProps {
  type?: "error" | "warning" | "info";
  title: string;
  message: string;
  nextStep?: string;
  onClose?: () => void;
}

export const NoticeBanner: React.FC<NoticeBannerProps> = ({
  type = "error",
  title,
  message,
  nextStep,
  onClose,
}) => {
  const styles = {
    error: {
      bg: "bg-status-rejected-bg",
      border: "border-status-rejected/30",
      text: "text-status-rejected",
      icon: AlertCircle,
    },
    warning: {
      bg: "bg-status-uncertain-bg",
      border: "border-status-uncertain/30",
      text: "text-status-uncertain",
      icon: AlertTriangle,
    },
    info: {
      bg: "bg-primary-light",
      border: "border-primary/30",
      text: "text-primary",
      icon: Info,
    },
  }[type];

  const IconComponent = styles.icon;

  return (
    <div
      role="alert"
      className={`w-full p-4 rounded-xl border ${styles.bg} ${styles.border} flex items-start justify-between gap-3 shadow-sm`}
    >
      <div className="flex items-start gap-3">
        <IconComponent className={`w-5 h-5 ${styles.text} flex-shrink-0 mt-0.5`} />
        <div className="space-y-1">
          <h4 className="text-sm font-semibold text-ink">{title}</h4>
          <p className="text-sm text-muted">{message}</p>
          {nextStep && (
            <p className="text-xs text-ink/80 pt-1">
              <strong>Next step:</strong> {nextStep}
            </p>
          )}
        </div>
      </div>
      {onClose && (
        <button
          onClick={onClose}
          className="text-muted hover:text-ink p-1 rounded-md transition-colors"
          aria-label="Close notification"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};
