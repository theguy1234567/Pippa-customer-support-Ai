export interface IntentResult {
  name: string;
  confidence: number;
  probabilities: Record<string, number>;
  is_actionable: boolean;
  method: string;
}

export interface EvidenceItem {
  case_id: string;
  tweet_id: string;
  conversation_id: string;
  role: "customer" | "support";
  timestamp: string | null;
  similarity: number;
  customer_message: string;
  support_response: string;
  resolution: string | null;
  source: string;
}

export interface DecisionResult {
  action: "AUTO_HANDLE" | "HUMAN_ESCALATION";
  confidence: number;
  reason: string;
  reason_codes: string[];
}

export interface SupportResponse {
  message: string;
  intent: IntentResult;
  evidence: EvidenceItem[];
  response: string | null;
  reply: string | null;
  generation_error: string | null;
  decision: DecisionResult;
  grounded: boolean;
  retrieval_method: string;
  generation_method: string;
}

export interface HealthResponse {
  status: string;
  classifier_loaded: boolean;
  retriever_loaded: boolean;
  llm_available: boolean;
}
