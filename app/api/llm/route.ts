import { NextRequest, NextResponse } from 'next/server'
import OpenAI from 'openai'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || '',
})

export async function POST(request: NextRequest) {
  try {
    const { faceCount, expressions } = await request.json()

    if (!process.env.OPENAI_API_KEY) {
      // Fallback response if no API key is set
      return NextResponse.json({
        message: `Detected ${faceCount} face(s) with expressions: ${expressions.join(', ')}. To enable AI analysis, please set your OPENAI_API_KEY in the .env.local file.`,
      })
    }

    const prompt = `You detected ${faceCount} face(s) in an image. The detected facial expressions are: ${expressions.join(', ')}. 

Provide a brief, friendly analysis of what you observe about the faces and their expressions. Keep it concise (2-3 sentences).`

    const completion = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [
        {
          role: 'system',
          content: 'You are a helpful AI assistant that analyzes facial expressions detected in images. Provide friendly, concise observations.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
      max_tokens: 150,
    })

    const message = completion.choices[0]?.message?.content || 'Unable to generate analysis.'

    return NextResponse.json({ message })
  } catch (error) {
    console.error('LLM API error:', error)
    return NextResponse.json(
      { error: 'Failed to process with LLM' },
      { status: 500 }
    )
  }
}


