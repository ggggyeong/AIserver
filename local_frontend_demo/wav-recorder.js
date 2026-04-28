export class WavRecorder {
  constructor() {
    this.audioContext = null;
    this.source = null;
    this.processor = null;
    this.stream = null;
    this.buffers = [];
    this.sampleRate = 44100;
    this.isRecording = false;
  }

  async start(stream) {
    if (this.isRecording) return;
    this.stream = stream;

    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    this.audioContext = new AudioContextCtor();
    this.sampleRate = this.audioContext.sampleRate;

    this.source = this.audioContext.createMediaStreamSource(stream);
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

    this.buffers = [];
    this.processor.onaudioprocess = (event) => {
      if (!this.isRecording) return;
      const channelData = event.inputBuffer.getChannelData(0);
      this.buffers.push(new Float32Array(channelData));
    };

    this.source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);
    this.isRecording = true;
  }

  async stop() {
    if (!this.isRecording) return null;
    this.isRecording = false;

    if (this.source) this.source.disconnect();
    if (this.processor) this.processor.disconnect();

    const wavBlob = this._encodeWavBlob(this.buffers, this.sampleRate);

    if (this.audioContext) {
      await this.audioContext.close();
    }

    this.audioContext = null;
    this.source = null;
    this.processor = null;

    return wavBlob;
  }

  _mergeBuffers(buffers) {
    let totalLength = 0;
    for (const buf of buffers) totalLength += buf.length;
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (const buf of buffers) {
      result.set(buf, offset);
      offset += buf.length;
    }
    return result;
  }

  _floatTo16BitPCM(view, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      let sample = Math.max(-1, Math.min(1, input[i]));
      sample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(offset, sample, true);
    }
  }

  _writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  _encodeWavBlob(buffers, sampleRate) {
    const samples = this._mergeBuffers(buffers);
    const bytesPerSample = 2;
    const numChannels = 1;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    this._writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    this._writeString(view, 8, "WAVE");
    this._writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    this._writeString(view, 36, "data");
    view.setUint32(40, dataSize, true);
    this._floatTo16BitPCM(view, 44, samples);

    return new Blob([view], { type: "audio/wav" });
  }
}
