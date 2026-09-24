// Silero VAD v5 (2 MB) on the CPU/wasm backend: ~0.1 ms per 32 ms frame.
import { AutoModel, Tensor } from '@huggingface/transformers';

type Callable = (inputs: Record<string, Tensor>) => Promise<Record<string, Tensor>>;

export class SileroVad {
  private model: Callable;
  private state: Tensor;
  private readonly sr = new Tensor('int64', [16000n], []);

  private constructor(model: Callable) {
    this.model = model;
    this.state = SileroVad.zeroState();
  }

  private static zeroState() {
    return new Tensor('float32', new Float32Array(2 * 128), [2, 1, 128]);
  }

  static async load(path: string, device: string): Promise<SileroVad> {
    const model = await AutoModel.from_pretrained(path, {
      config: { model_type: 'custom' } as never,
      dtype: 'fp32',
      device: device as never,
    });
    return new SileroVad(model as unknown as Callable);
  }

  /** `frame` is 64 context + 512 new samples at 16 kHz. */
  async prob(frame: Float32Array): Promise<number> {
    const input = new Tensor('float32', frame, [1, frame.length]);
    const out = await this.model({ input, sr: this.sr, state: this.state });
    this.state = out.stateN;
    return (out.output.data as Float32Array)[0];
  }

  reset() {
    this.state = SileroVad.zeroState();
  }
}
