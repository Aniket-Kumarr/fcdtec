import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Face Detection AI',
  description: 'Face detection with LLM processing',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}


