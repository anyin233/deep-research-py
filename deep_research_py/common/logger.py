import logging

logger = logging.getLogger("deep_research_py")
logger_enabled = False

def setup_logger(log_file: str = "deep_research_py.log") -> None:
  """Sets up the logger for the application."""
  global logger_enabled
  logger_enabled = True
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      handlers=[
          logging.StreamHandler(),
          logging.FileHandler(log_file),
      ],
  )
    
def log_event(event_desc: str, input_tokens: int, output_tokens: int) -> None:
  """Logs an event with input and output token counts."""
  if not logger_enabled:
      return
  logger.info(
      f"Event: {event_desc}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}"
  )
  
def log_error(error_desc: str) -> None:
  """Logs an error message."""
  if not logger_enabled:
      return
  logger.error(f"Error: {error_desc}")

