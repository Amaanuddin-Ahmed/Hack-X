"use client"

import * as React from "react"
import { Progress as ProgressPrimitive } from "radix-ui"
import { animate, motion, useMotionValue, useTransform } from "motion/react"

import { cn } from "@/lib/utils"

function Progress({
  className,
  value,
  ...props
}: React.ComponentProps<typeof ProgressPrimitive.Root>) {
  return (
    <ProgressPrimitive.Root
      data-slot="progress"
      value={value}
      className={cn(
        "relative h-2 w-full overflow-hidden rounded-full bg-primary/20",
        className
      )}
      {...props}
    >
      <ProgressPrimitive.Indicator
        data-slot="progress-indicator"
        className="h-full w-full flex-1 bg-primary transition-all"
        style={{ transform: `translateX(-${100 - (value ?? 0)}%)` }}
      />
    </ProgressPrimitive.Root>
  )
}

interface Vo2MaxCardProps {
  title: string
  value: number
  status: string
  description: React.ReactNode
  progress: number
  icon: React.ReactNode
  className?: string
}

const Vo2MaxCard: React.FC<Vo2MaxCardProps> = ({
  title,
  value,
  status,
  description,
  progress,
  icon,
  className,
}) => {
  const count = useMotionValue(0)
  const rounded = useTransform(count, latest => Math.round(latest))
  const progressValue = useMotionValue(0)

  React.useEffect(() => {
    const valueAnimation = animate(count, value, {
      duration: 1.5,
      ease: [0.43, 0.13, 0.23, 0.96],
    })
    const progressAnimation = animate(progressValue, progress, {
      duration: 1.5,
      ease: [0.43, 0.13, 0.23, 0.96],
    })
    return () => {
      valueAnimation.stop()
      progressAnimation.stop()
    }
  }, [value, progress, count, progressValue])

  const radius = 80
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = useTransform(
    progressValue,
    current => circumference - (current / 100) * circumference
  )

  return (
    <section className={cn("health-score-card card", className)}>
      <div className="health-score-card__header">
        <h3>{title}</h3>
        <div className="health-score-card__icon">{icon}</div>
      </div>
      <div className="health-score-card__radial">
        <svg width="200" height="200" viewBox="0 0 200 200" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
          <circle cx="100" cy="100" r={radius} strokeWidth="12" fill="transparent" className="health-score-card__track" strokeDasharray="8 12" strokeLinecap="round" />
          <motion.circle cx="100" cy="100" r={radius} strokeWidth="12" fill="transparent" className="health-score-card__progress" strokeDasharray={`${circumference} ${circumference}`} strokeLinecap="round" style={{ strokeDashoffset }} />
        </svg>
        <div className="health-score-card__value">
          <motion.span>{rounded}</motion.span>
          <p>{status}</p>
        </div>
      </div>
      <div className="health-score-card__description">{description}</div>
    </section>
  )
}

export { Progress, Vo2MaxCard }
