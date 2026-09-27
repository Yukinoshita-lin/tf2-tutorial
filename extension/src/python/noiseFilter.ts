/**
 * 无害日志过滤（纯逻辑，可单元测试）。
 * TF 导入时 absl/oneDNN/CPU 特性提示会吓到初学者——显示层隐藏，
 * 完整 stderr 仍保留供报错翻译匹配。
 */

export const NOISE_RE =
  /absl::InitializeLog|oneDNN custom operations|cpu_feature_guard|To enable the following instructions/;

export interface NoiseFilterResult {
  /** 应当显示的行（各自带换行语义由调用方决定，这里返回不含换行的行内容） */
  shown: string[];
  /** 本批次隐藏的噪音行数 */
  noise: number;
  /** 因缺换行而无法判定归属的半行（下一批继续拼接） */
  carry: string;
}

export interface NoiseFilterState {
  carry: string;
  total: number;
  notified: boolean;
}

export function initialState(): NoiseFilterState {
  return { carry: "", total: 0, notified: false };
}

/**
 * 处理一段 stderr 文本（可能以半行结尾）。
 * 返回应显示的完整行 + 新状态。半行保留在 carry，下次拼接。
 */
export function filterNoise(state: NoiseFilterState, text: string): NoiseFilterResult {
  const parts = (state.carry + text).split("\n");
  const carry = parts.pop() ?? "";
  const shown: string[] = [];
  let noise = 0;
  for (const line of parts) {
    if (NOISE_RE.test(line)) {
      noise++;
      if (!state.notified) {
        state.notified = true;
        shown.push("[TF 学习伴侣] （已隐藏 TensorFlow 初始化的无害日志：absl / oneDNN）");
      }
    } else {
      shown.push(line);
    }
  }
  state.carry = carry;
  state.total += noise;
  return { shown, noise, carry };
}

/** 流结束时处理 carry 尾巴：若是噪音则吞掉，否则显示 */
export function flushNoise(state: NoiseFilterState): { shown: string[]; noise: number } {
  const result: NoiseFilterResult = { shown: [], noise: 0, carry: "" };
  if (state.carry) {
    if (NOISE_RE.test(state.carry)) {
      state.total++;
      result.noise = 1;
    } else {
      result.shown.push(state.carry);
    }
    state.carry = "";
  }
  return result;
}
