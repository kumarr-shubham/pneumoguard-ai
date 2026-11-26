export type PredictionResult = {
  label: 'NORMAL' | 'PNEUMONIA';
  confidence: number;
  reasoning: string;
  clinicalNotes: string;
  explanation?: string;      // Gemini-generated explanation (optional, may fail gracefully)
  confidencePercent?: number; // Raw % from MobileNetV2
};

export type ReportData = {
  patientName: string;
  date: string;
  prediction: PredictionResult;
  imageUrl: string;
};