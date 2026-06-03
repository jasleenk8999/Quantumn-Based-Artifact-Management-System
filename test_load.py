from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM

_original_dense_init = Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_dense_init(self, *args, **kwargs)
Dense.__init__ = _patched_dense_init

_original_lstm_init = LSTM.__init__
def _patched_lstm_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_lstm_init(self, *args, **kwargs)
LSTM.__init__ = _patched_lstm_init

try:
    model = load_model('models/future_score_model.keras')
    print('SUCCESS_LOAD')
except Exception as e:
    print('ERROR:', e)
