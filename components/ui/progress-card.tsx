"use client"

import * as React from "react"
import { motion } from "motion/react"
import { cn } from "@/lib/utils"

export interface AnimatedProgressCardProps {
  icon: React.ReactNode
  title: string
  progressLabel: string
  progressSubLabel: string
  currentValue: number
  maxValue: number
  className?: string
}

export const AnimatedProgressCard = React.forwardRef<HTMLDivElement, AnimatedProgressCardProps>(
  ({ icon, title, progressLabel, progressSubLabel, currentValue, maxValue, className }, ref) => {
    const percentage = maxValue > 0 ? (currentValue / maxValue) * 100 : 0
    const clampedPercentage = Math.min(Math.max(percentage, 0), 100)

    return (
      <div ref={ref} className={cn("animated-progress-card", className)}>
        <div className="animated-progress-card__header">
          <div className="animated-progress-card__icon">{icon}</div>
          <p>{title}</p>
        </div>
        <div className="animated-progress-card__bar" role="progressbar" aria-valuenow={currentValue} aria-valuemin={0} aria-valuemax={maxValue} aria-label={title}>
          <motion.div initial={{ width: 0 }} animate={{ width: `${clampedPercentage}%` }} transition={{ duration: 1.2, ease: "easeInOut" }}/>
        </div>
        <div className="animated-progress-card__footer">
          <div>
            <p>{progressLabel}</p>
            <span>{progressSubLabel}</span>
          </div>
          <strong>{currentValue}<small> / {maxValue}</small></strong>
        </div>
      </div>
    )
  }
)

AnimatedProgressCard.displayName = "AnimatedProgressCard"
