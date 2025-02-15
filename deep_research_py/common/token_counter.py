from dataclasses import dataclass

@dataclass
class TokenEvent:
  event: str
  input_tokens: int
  output_tokens: int = 0
  accepted_prediction_tokens: int = 0
  rejected_prediction_tokens: int = 0
  reasoning_token: int = 0
  
  def __repr__(self):
    return f"TokenEvent(event='{self.event}', input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, accepted_prediction_tokens={self.accepted_prediction_tokens}, rejected_prediction_tokens={self.rejected_prediction_tokens}, reasoning_token={self.reasoning_token})"

@dataclass
class TokenCounter:
  input_tokens: int = 0
  output_tokens: int = 0
  events: list[TokenEvent] = None
  
  def __post_init__(self):
    self.events = []
  
  def add_event(self, event: TokenEvent):
    self.events.append(event)
    self.input_tokens += event.input_tokens
    self.output_tokens += event.output_tokens

  def __repr__(self):
    return f"TokenCounter(input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, events={self.events})"
  
def count_token(response, event_desc):
  input_token = response.usage.prompt_tokens
  output_token = response.usage.completion_tokens
  if hasattr(response.usage, "completion_tokens_details"):
    reasoning_token = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0)
    accepted_prediction_tokens = getattr(response.usage.completion_tokens_details, 'accepted_prediction_tokens', 0)
    rejected_prediction_tokens = getattr(response.usage.completion_tokens_details, 'rejected_prediction_tokens', 0)
  event = TokenEvent(
      event=event_desc,
      input_tokens=input_token,
      output_tokens=output_token,
      accepted_prediction_tokens=accepted_prediction_tokens,
      rejected_prediction_tokens=rejected_prediction_tokens,
      reasoning_token=reasoning_token,
  )
  counter.add_event(event)
        
counter = TokenCounter()