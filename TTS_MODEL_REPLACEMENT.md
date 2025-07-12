# 🎵 TTS Model Replacement Guide

## 📋 Overview

This guide documents how to replace the current **SpeechT5** TTS model with alternative models (MMS, Bark, XTTS, etc.) in the CRAG Assistant project.

## 🎯 Current Implementation

The project currently uses **SpeechT5** from Hugging Face Transformers for English text-to-speech generation.

---

## 📁 Files and Methods to Modify

### **File: `crag_api.py`**

#### **1. TTSService Class - Model Loading**

**Method: `_load_model()`** (Lines ~90-130)

**Current SpeechT5 Implementation:**
```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# Model initialization
self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
```

**Replacement Required:**
- Update import statements for new model
- Change model loading logic
- Modify processor initialization

**Examples for Different Models:**
```python
# For MMS TTS:
from transformers import VitsModel, AutoTokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

# For Bark:
from bark import SAMPLE_RATE, generate_audio, preload_models
preload_models()

# For XTTS:
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
```

#### **2. Speaker Embeddings Generation**

**Method: `_create_speaker_embedding()`** (Lines ~150-180)

**Current Implementation:**
```python
def _create_speaker_embedding(self):
    # Creates 512-dimensional embedding for SpeechT5
    embedding = np.zeros(512)
    # ... specific female voice characteristics
    return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
```

**Modification Requirements:**
- **MMS**: Remove speaker embeddings (not needed)
- **Bark**: Replace with text-based speaker prompts
- **XTTS**: Use different embedding format or voice cloning samples

#### **3. Audio Generation Logic**

**Method: `_generate_audio_sync()`** (Lines ~200-250)

**Current SpeechT5 Generation:**
```python
# Tokenize text
inputs = self.processor(text=text, return_tensors="pt")

# Generate speech
with torch.no_grad():
    speech = self.model.generate_speech(
        inputs["input_ids"], 
        self.speaker_embeddings, 
        vocoder=self.vocoder
    )

# Convert to numpy
audio_data = speech.cpu().numpy()
```

**Replacement Areas:**
- Text tokenization/preprocessing
- Model inference call
- Output format handling
- Post-processing pipeline

#### **4. Text Processing**

**Method: `_clean_text()`** (Lines ~270-290)

**Current Limitations:**
```python
# SpeechT5 specific constraints
if len(text) > 500:
    text = text[:500] + "..."
```

**Update Requirements:**
- Model-specific text length limits
- Language-specific preprocessing
- Special character handling

---

## 📋 Configuration Changes

### **File: `requirements.txt`**

**Current TTS Dependencies:**
```
torch==2.1.2+cpu
transformers>=4.35.0
soundfile>=0.12.1
librosa>=0.10.1
sentencepiece>=0.1.99
```

**Model-Specific Dependencies:**

**For MMS:**
```
transformers>=4.35.0
phonemizer>=3.2.0
espeak-ng
```

**For Bark:**
```
bark>=1.0.0
encodec>=0.1.0
scipy>=1.10.0
```

**For XTTS:**
```
TTS>=0.20.0
espeak-ng
phonemizer>=3.2.0
```

### **File: `.env` (Environment Variables)**

**Add Model-Specific Configuration:**
```env
# TTS Model Configuration
TTS_MODEL_TYPE=speecht5  # speecht5, mms, bark, xtts
TTS_MODEL_NAME=microsoft/speecht5_tts

# Model-specific settings
TTS_LANGUAGE=en          # For multilingual models
TTS_VOICE_PRESET=female  # For models with voice presets
TTS_SAMPLE_RATE=16000    # Model-specific sample rate
```

---

## 📊 Audio Parameter Updates

### **Sample Rate Considerations**

```python
# Current (SpeechT5):
sf.write(str(audio_path), audio_final, 16000)

# Model-specific sample rates:
# MMS: 16000 Hz
# Bark: 24000 Hz  
# XTTS: 22050 Hz
# Commercial APIs: 24000-48000 Hz
```

### **Audio Quality Post-Processing**

**Current Pipeline:**
```python
# Time stretching for speed control
audio_slowed = librosa.effects.time_stretch(audio_data, rate=0.75)

# Volume normalization
audio_normalized = librosa.util.normalize(audio_slowed)
audio_boosted = audio_normalized * 1.25
audio_final = np.clip(audio_boosted, -1.0, 1.0)
```

**Considerations for New Models:**
- Some models may not need speed adjustment
- Different models may have different volume characteristics
- Quality enhancement parameters may need tuning

---

## 🛠️ Common Issues and Solutions

### **Memory Issues**
```python
# Problem: New model uses too much memory
# Solution: Model optimization
import torch
model.half()  # Use 16-bit precision
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
```

### **Speed Issues**
```python
# Problem: Generation too slow
# Solutions:
# 1. Use smaller model variant
# 2. Implement model compilation
# 3. Batch processing for multiple requests
```

### **Quality Issues**
```python
# Problem: Poor audio quality
# Solutions:
# 1. Adjust generation parameters
# 2. Use higher sample rate
# 3. Implement better post-processing
# 4. Try different model variants
```

### **Compatibility Issues**
```python
# Problem: Model doesn't work with current pipeline
# Solutions:
# 1. Update text preprocessing
# 2. Modify audio format handling
# 3. Adjust API response format
# 4. Update frontend audio player
```

---

## 📞 Support and Troubleshooting

### **When Migrating to New TTS Model:**

1. **Preserve current functionality**: Keep existing TTS working during migration
2. **Document all changes**: Note what works and what doesn't
3. **Test incrementally**: Don't replace everything at once
4. **Have rollback plan**: Keep SpeechT5 code as backup
5. **Monitor performance**: Compare before/after metrics

### **Key Files to Monitor:**
- `crag_api.py` - Main TTS implementation
- `requirements.txt` - Dependencies
- `.env` - Configuration
- `audio_files/` - Output directory
- Frontend audio player - Compatibility

---

## 🔗 Useful Resources

### **Model Documentation:**
- **SpeechT5**: https://huggingface.co/microsoft/speecht5_tts
- **MMS**: https://huggingface.co/facebook/mms-tts
- **Bark**: https://github.com/suno-ai/bark
- **XTTS**: https://github.com/coqui-ai/TTS

### **Audio Processing:**
- **Librosa**: https://librosa.org/doc/latest/
- **SoundFile**: https://python-soundfile.readthedocs.io/
- **PyTorch Audio**: https://pytorch.org/audio/stable/

---

**📋 This guide provides complete coverage of all SpeechT5-specific code that requires modification when replacing the TTS model in the CRAG Assistant project.**