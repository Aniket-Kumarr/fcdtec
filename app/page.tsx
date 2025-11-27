'use client'

import { useState, useRef, useEffect } from 'react'
import * as faceapi from 'face-api.js'

export default function Home() {
  const [image, setImage] = useState<string | null>(null)
  const [detections, setDetections] = useState<faceapi.WithFaceExpressions<faceapi.WithFaceLandmarks<faceapi.WithFaceDescriptor<faceapi.FaceDetection>>>[]>([])
  const [llmResponse, setLlmResponse] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [isWebcamActive, setIsWebcamActive] = useState(false)
  const [mode, setMode] = useState<'upload' | 'webcam'>('upload')
  
  const imageRef = useRef<HTMLImageElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Load face-api.js models
  useEffect(() => {
    const loadModels = async () => {
      try {
        const MODEL_URL = '/models'
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        ])
        setModelsLoaded(true)
      } catch (err) {
        setError('Failed to load face detection models. Please ensure models are in the public/models directory.')
        console.error('Error loading models:', err)
      }
    }
    loadModels()
  }, [])

  // Draw detections on canvas
  useEffect(() => {
    if (detections.length > 0 && canvasRef.current) {
      const canvas = canvasRef.current
      let displaySize: { width: number; height: number }
      
      if (mode === 'webcam' && videoRef.current) {
        displaySize = { width: videoRef.current.videoWidth, height: videoRef.current.videoHeight }
      } else if (mode === 'upload' && imageRef.current) {
        displaySize = { width: imageRef.current.width, height: imageRef.current.height }
      } else {
        return
      }
      
      faceapi.matchDimensions(canvas, displaySize)
      
      const resizedDetections = faceapi.resizeResults(detections, displaySize)
      
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        
        // Draw detections
        faceapi.draw.drawDetections(canvas, resizedDetections)
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
        faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
      }
    }
  }, [image, detections, mode])

  // Real-time face detection for webcam
  useEffect(() => {
    if (isWebcamActive && videoRef.current && modelsLoaded && canvasRef.current) {
      const video = videoRef.current
      const canvas = canvasRef.current
      
      const detectFacesInVideo = async () => {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
          const displaySize = { width: video.videoWidth, height: video.videoHeight }
          faceapi.matchDimensions(canvas, displaySize)
          
          const detections = await faceapi
            .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceDescriptors()
            .withFaceExpressions()
          
          setDetections(detections)
          
          const resizedDetections = faceapi.resizeResults(detections, displaySize)
          const ctx = canvas.getContext('2d')
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height)
            faceapi.draw.drawDetections(canvas, resizedDetections)
            faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
            faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
          }
        }
      }
      
      detectionIntervalRef.current = setInterval(detectFacesInVideo, 100)
      
      return () => {
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current)
        }
      }
    }
  }, [isWebcamActive, modelsLoaded])

  // Cleanup webcam on unmount
  useEffect(() => {
    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current)
      }
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  const handleImageUpload = async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file')
      return
    }

    setError('')
    setLlmResponse('')
    setDetections([])

    const reader = new FileReader()
    reader.onload = (e) => {
      setImage(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const detectFaces = async () => {
    if (!image || !imageRef.current || !modelsLoaded) return

    setLoading(true)
    setError('')

    try {
      const img = imageRef.current
      const detections = await faceapi
        .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors()
        .withFaceExpressions()

      setDetections(detections)

      // Send to LLM for processing
      if (detections.length > 0) {
        await processWithLLM(detections)
      } else {
        setLlmResponse('No faces detected in the image.')
      }
    } catch (err) {
      setError('Failed to detect faces: ' + (err instanceof Error ? err.message : 'Unknown error'))
      console.error('Detection error:', err)
    } finally {
      setLoading(false)
    }
  }

  const processWithLLM = async (faces: typeof detections) => {
    try {
      const expressions = faces.map(face => {
        const expressions = face.expressions
        const topExpression = Object.entries(expressions)
          .sort(([, a], [, b]) => b - a)[0]
        return topExpression[0]
      })

      const response = await fetch('/api/llm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          faceCount: faces.length,
          expressions: expressions,
        }),
      })

      if (!response.ok) {
        throw new Error('LLM processing failed')
      }

      const data = await response.json()
      setLlmResponse(data.message)
    } catch (err) {
      setError('Failed to process with LLM: ' + (err instanceof Error ? err.message : 'Unknown error'))
      console.error('LLM error:', err)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) {
      handleImageUpload(file)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const startWebcam = async () => {
    try {
      setError('')
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current && canvasRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth
            canvasRef.current.height = videoRef.current.videoHeight
            videoRef.current.play()
            setIsWebcamActive(true)
            setMode('webcam')
            setImage(null)
            setLlmResponse('')
          }
        }
      }
    } catch (err) {
      setError('Failed to access webcam: ' + (err instanceof Error ? err.message : 'Unknown error'))
      console.error('Webcam error:', err)
    }
  }

  const stopWebcam = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
      detectionIntervalRef.current = null
    }
    setIsWebcamActive(false)
    setDetections([])
    setLlmResponse('')
  }

  const handleModeSwitch = (newMode: 'upload' | 'webcam') => {
    if (newMode === 'webcam' && !isWebcamActive) {
      // Clear upload image when switching to webcam
      setImage(null)
      setDetections([])
      setLlmResponse('')
      startWebcam()
    } else if (newMode === 'upload' && isWebcamActive) {
      stopWebcam()
      setMode('upload')
    } else {
      setMode(newMode)
    }
  }

  const captureAndAnalyze = async () => {
    if (!videoRef.current || !modelsLoaded) return

    setLoading(true)
    setError('')

    try {
      const detections = await faceapi
        .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors()
        .withFaceExpressions()

      if (detections.length > 0) {
        await processWithLLM(detections)
      } else {
        setLlmResponse('No faces detected.')
      }
    } catch (err) {
      setError('Failed to analyze: ' + (err instanceof Error ? err.message : 'Unknown error'))
      console.error('Analysis error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>Face Detection AI</h1>
      
      {error && <div className="error">{error}</div>}

      {/* Mode Toggle */}
      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginBottom: '1.5rem' }}>
        <button
          className={`button ${mode === 'upload' ? 'active' : ''}`}
          onClick={() => handleModeSwitch('upload')}
          style={{
            background: mode === 'upload' ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : '#e0e0e0',
            color: mode === 'upload' ? 'white' : '#666',
          }}
        >
          Upload Image
        </button>
        <button
          className={`button ${mode === 'webcam' ? 'active' : ''}`}
          onClick={() => handleModeSwitch('webcam')}
          style={{
            background: mode === 'webcam' ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : '#e0e0e0',
            color: mode === 'webcam' ? 'white' : '#666',
          }}
        >
          Webcam
        </button>
      </div>

      {mode === 'upload' ? (
        <>
          <div
            className={`upload-area ${image ? '' : 'dragover'}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current?.click()}
          >
            {image ? (
              <div className="image-container">
                <img
                  ref={imageRef}
                  src={image}
                  alt="Uploaded"
                  onLoad={() => {
                    if (canvasRef.current && imageRef.current) {
                      canvasRef.current.width = imageRef.current.width
                      canvasRef.current.height = imageRef.current.height
                    }
                  }}
                />
                <canvas
                  ref={canvasRef}
                  className="canvas-overlay"
                />
              </div>
            ) : (
              <div>
                <p>Click or drag an image here to upload</p>
                <p style={{ marginTop: '0.5rem', color: '#666', fontSize: '0.9rem' }}>
                  Supports JPG, PNG, and other image formats
                </p>
              </div>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) {
                handleImageUpload(file)
              }
            }}
          />

          {image && (
            <div style={{ textAlign: 'center', marginTop: '1rem' }}>
              <button
                className="button"
                onClick={detectFaces}
                disabled={loading || !modelsLoaded}
              >
                {loading ? 'Processing...' : 'Detect Faces'}
              </button>
              {detections.length > 0 && (
                <p style={{ marginTop: '1rem', color: '#666' }}>
                  Detected {detections.length} face{detections.length !== 1 ? 's' : ''}
                </p>
              )}
            </div>
          )}
        </>
      ) : (
        <>
          <div className="image-container" style={{ position: 'relative', display: 'inline-block' }}>
            {isWebcamActive ? (
              <>
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  style={{
                    maxWidth: '100%',
                    height: 'auto',
                    border: '2px solid #667eea',
                    borderRadius: '8px',
                    transform: 'scaleX(-1)', // Mirror the video
                    display: 'block',
                  }}
                />
                <canvas
                  ref={canvasRef}
                  className="canvas-overlay"
                  style={{
                    transform: 'scaleX(-1)', // Mirror the canvas to match video
                  }}
                />
              </>
            ) : (
              <div className="upload-area" style={{ minHeight: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div>
                  <p>Click "Start Webcam" to begin</p>
                  <button
                    className="button"
                    onClick={startWebcam}
                    disabled={!modelsLoaded}
                    style={{ marginTop: '1rem' }}
                  >
                    Start Webcam
                  </button>
                </div>
              </div>
            )}
          </div>

          {isWebcamActive && (
            <div style={{ textAlign: 'center', marginTop: '1rem' }}>
              <button
                className="button"
                onClick={stopWebcam}
                style={{ background: '#dc3545' }}
              >
                Stop Webcam
              </button>
              <button
                className="button"
                onClick={captureAndAnalyze}
                disabled={loading || !modelsLoaded}
              >
                {loading ? 'Analyzing...' : 'Analyze with AI'}
              </button>
              {detections.length > 0 && (
                <p style={{ marginTop: '1rem', color: '#666' }}>
                  Detected {detections.length} face{detections.length !== 1 ? 's' : ''}
                </p>
              )}
            </div>
          )}
        </>
      )}

      {!modelsLoaded && (
        <div className="loading">
          Loading face detection models...
        </div>
      )}

      {llmResponse && (
        <div className="llm-output">
          <h3>AI Analysis</h3>
          <p>{llmResponse}</p>
        </div>
      )}
    </div>
  )
}

